from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPTextModel, SiglipTextModel
from .sam3_base import SAM3Wrapper
from .cross_attn_fusion import CrossAttentionFusion

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:  # pragma: no cover - handled at runtime with a clear message
    LoraConfig = None
    PeftModel = None
    get_peft_model = None

try:
    from sam3.model.data_misc import FindStage, interpolate
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SAM3 must be importable. Add the SAM3 checkout to PYTHONPATH or VSCode"
        "extraPaths before running this pipeline."
    ) from exc

def select_text_conditioned_masks(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Select the highest-scoring predicted mask for each batch item."""

    logits = out["pred_logits"].squeeze(-1)
    best_idx = logits.argmax(dim=1)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    return out["pred_masks"][batch_idx, best_idx]


def filter_text_conditioned_masks(
        out: Dict[str, torch.Tensor],
        score_threshold: float,
    ) -> torch.Tensor:
    """
    Return all predicted mask logits above the object-score threshold.

    Falls back to the highest-scoring mask when no candidate passes the threshold.
    This keeps inference useful on poorly calibrated checkpoints while allowing
    generic prompts to return multiple object masks.
    """

    mask_logits = out["pred_masks"][0]
    if mask_logits.ndim == 2:
        mask_logits = mask_logits.unsqueeze(0)
    elif mask_logits.ndim == 4 and mask_logits.shape[1] == 1:
        mask_logits = mask_logits.squeeze(1)

    scores = out.get("pred_logits")
    if scores is None:
        return mask_logits

    scores = scores[0].squeeze(-1).sigmoid()
    keep = scores > score_threshold
    if not keep.any():
        keep[scores.argmax()] = True
    return mask_logits[keep]

class TextConditionedSAM3LoRA(nn.Module):
    """SAM3 wrapper that trains LoRA adapters and a text fusion module."""

    def __init__(self, sam3: SAM3Wrapper, config: Dict[str, Any]) -> None:
        """Build PEFT-wrapped SAM3 modules and the external text encoder."""

        super().__init__()
        self.sam3_wrapper = sam3
        self.sam3 = sam3.model
        self.config = config
        self.device_name = config.get("device", "cuda")
        self.resolution = config.get("data", {}).get("resolution", 1008)
        self.confidence_threshold = config.get("inference", {}).get(
            "confidence_threshold", 0.5
        )
        self.mask_score_threshold = config.get("inference", {}).get(
            "mask_score_threshold", self.confidence_threshold
        )

        self._freeze_sam3_base()
        self._apply_lora(config.get("lora", {}))
        self.tokenizer, self.text_encoder, text_dim = self._build_text_encoder(
            config.get("text_encoder", {})
        )
        self.fusion = CrossAttentionFusion(
            sam_dim=self.sam3.hidden_dim,
            text_dim=text_dim,
            num_heads=config.get("fusion", {}).get("num_heads", 8),
        )

    def _freeze_sam3_base(self) -> None:
        """Freeze all SAM3 base parameters before applying trainable LoRA adapters."""

        for param in self.sam3.parameters():
            param.requires_grad = False

    def _require_peft(self) -> None:
        """Raise a clear error when PEFT is unavailable."""

        if get_peft_model is None:
            raise ImportError(
                "HuggingFace PEFT is required for LoRA. Install it with "
                "`pip install peft` in this environment."
            )

    def _apply_lora(self, cfg: Dict[str, Any]) -> None:
        """Attach LoRA adapters to SAM3 vision and transformer decoder modules."""

        if not cfg.get("enabled", True):
            return
        self._require_peft()
        target_modules = cfg.get(
            "target_modules", ["qkv", "proj", "out_proj", "linear1", "linear2"]
        )
        lora_cfg = LoraConfig(
            r=cfg.get("r", 8),
            lora_alpha=cfg.get("alpha", 16),
            lora_dropout=cfg.get("dropout", 0.05),
            bias=cfg.get("bias", "none"),
            target_modules=target_modules,
        )
        self.sam3.backbone.vision_backbone = get_peft_model(
            self.sam3.backbone.vision_backbone, lora_cfg
        )
        self.sam3.transformer.decoder = get_peft_model(
            self.sam3.transformer.decoder, lora_cfg
        )

    def _build_text_encoder(
        self, cfg: Dict[str, Any]
    ) -> Tuple[AutoTokenizer, nn.Module, int]:
        """Load a HuggingFace text encoder and return its hidden size."""

        model_name = cfg.get("name", "openai/clip-vit-base-patch32")
        self.freeze_text_encoder = cfg.get("freeze", True)
        model_kwargs = {"use_safetensors": cfg.get("use_safetensors", False)}
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_config = AutoConfig.from_pretrained(model_name)
        model_type = getattr(hf_config, "model_type", "")
        if model_type == "clip":
            text_encoder = CLIPTextModel.from_pretrained(model_name, **model_kwargs)
        elif model_type == "siglip":
            text_encoder = SiglipTextModel.from_pretrained(model_name, **model_kwargs)
        else:
            text_encoder = AutoModel.from_pretrained(model_name, **model_kwargs)
        if self.freeze_text_encoder:
            for param in text_encoder.parameters():
                param.requires_grad = False
            text_encoder.eval()
        hidden_size = getattr(
            text_encoder.config,
            "hidden_size",
            getattr(getattr(text_encoder.config, "text_config", None), "hidden_size", None),
        )
        if hidden_size is None:
            raise ValueError(f"Could not infer text hidden size for {model_name}.")
        return tokenizer, text_encoder, hidden_size

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return parameters optimized during LoRA/fusion training."""

        return [p for p in self.parameters() if p.requires_grad]

    def _external_text_tokens(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize object names and encode them with CLIP/SigLIP text features."""

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        with torch.set_grad_enabled(any(p.requires_grad for p in self.text_encoder.parameters())):
            output = self.text_encoder(**tokens)
        return output.last_hidden_state, tokens["attention_mask"]

    def forward(self, images: torch.Tensor, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Run SAM3 grounding with fused external text features.

        Args:
            images: Normalized image tensor of shape `[B, 3, H, W]`.
            texts: Object prompts, one per image.

        Returns:
            SAM3 output dictionary containing logits, boxes, masks, and internals.
        """

        device = images.device
        batch_size = images.shape[0]

        backbone_out = self.sam3.backbone.forward_image(images)
        sam_text = self.sam3.backbone.forward_text(texts, device=device)
        ext_tokens, ext_mask = self._external_text_tokens(texts, device)
        sam_text["language_features"] = self.fusion(
            sam_text["language_features"], ext_tokens, ext_mask
        )
        backbone_out.update(sam_text)

        find_input = FindStage(
            img_ids=torch.arange(batch_size, device=device, dtype=torch.long),
            text_ids=torch.arange(batch_size, device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geometric_prompt = self.sam3._get_dummy_prompt(num_prompts=batch_size)
        prompt, prompt_mask, backbone_out = self.sam3._encode_prompt(
            backbone_out, find_input, geometric_prompt
        )
        backbone_out, encoder_out, _ = self.sam3._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {"encoder_out": encoder_out, "backbone_out": backbone_out},
        }
        out, hs = self.sam3._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )
        self.sam3._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=find_input.img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
        )
        return out

    @torch.no_grad()
    def predict_mask(self, image: Image.Image, text: str, device: torch.device) -> torch.Tensor:
        """
        Predict candidate masks for an image/object prompt pair.

        Returns:
            Tensor of shape `[N, H, W]` containing one binary mask per kept
            prediction, resized to the input image size.
        """

        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(self.resolution, self.resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        width, height = image.size
        image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        out = self.forward(image_tensor, [text])
        logits = filter_text_conditioned_masks(out, self.mask_score_threshold)
        logits = interpolate(
            logits[:, None], (height, width), mode="bilinear", align_corners=False
        )[:, 0]
        return (logits.sigmoid() > self.confidence_threshold).detach().cpu()

    def save_checkpoint(self, output_dir: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save fusion weights, metrics, config, and separate LoRA adapter weights."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "fusion": self.fusion.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            output / "checkpoint.pt",
        )
        self.export_lora(output / "lora")

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True) -> Dict[str, Any]:
        """Load fusion-module checkpoint state and return checkpoint metadata."""

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.fusion.load_state_dict(checkpoint["fusion"], strict=strict)
        return checkpoint

    def load_lora(self, lora_dir: str) -> None:
        """Load exported image-encoder and decoder LoRA adapters."""

        if PeftModel is None:
            self._require_peft()
        base = Path(lora_dir)
        self._load_peft_adapter(
            self.sam3.backbone.vision_backbone, base / "image_encoder", "loaded_image"
        )
        self._load_peft_adapter(
            self.sam3.transformer.decoder, base / "mask_decoder", "loaded_decoder"
        )

    @staticmethod
    def _load_peft_adapter(module: nn.Module, adapter_dir: Path, adapter_name: str) -> None:
        """Load one PEFT adapter into an already PEFT-wrapped module."""

        if not adapter_dir.exists():
            raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_dir}")
        if hasattr(module, "load_adapter"):
            module.load_adapter(str(adapter_dir), adapter_name=adapter_name, is_trainable=False)
            module.set_adapter(adapter_name)
            return
        raise TypeError(
            f"Expected a PEFT-wrapped module with load_adapter(), got {type(module).__name__}"
        )

    def export_lora(self, output_dir: Path) -> None:
        """Export LoRA adapter weights without saving the SAM3 base checkpoint."""

        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.sam3.backbone.vision_backbone, "save_pretrained"):
            self.sam3.backbone.vision_backbone.save_pretrained(output_dir / "image_encoder")
        if hasattr(self.sam3.transformer.decoder, "save_pretrained"):
            self.sam3.transformer.decoder.save_pretrained(output_dir / "mask_decoder")
