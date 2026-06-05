import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPTextModel, SiglipTextModel

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
        "SAM3 must be importable. Add the SAM3 checkout to PYTHONPATH or VS Code "
        "extraPaths before running this pipeline."
    ) from exc

try:
    from .get_config import get_config
    from .segmentation import SegmentationModel
except ImportError:
    from get_config import get_config
    from segmentation import SegmentationModel


@dataclass
class Batch:
    images: torch.Tensor
    masks: torch.Tensor
    texts: List[str]
    image_paths: List[str]
    mask_paths: List[str]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_enabled(device: torch.device, requested: bool) -> bool:
    return requested and device.type in {"cuda", "cpu"}


def autocast_dtype(device: torch.device, requested: str) -> torch.dtype:
    if device.type == "cpu":
        return torch.bfloat16
    return torch.bfloat16 if requested == "bfloat16" else torch.float16


def resolve_path(path: str, base_dir: str = ".") -> str:
    p = Path(path)
    return str(p if p.is_absolute() else Path(base_dir) / p)


class DentalInstrumentDataset(Dataset):
    def __init__(
        self,
        annotation_path: str,
        image_dir: str,
        mask_dir: str,
        resolution: int = 1008,
        alias_mode: str = "random",
    ):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.resolution = resolution
        self.alias_mode = alias_mode

        with open(annotation_path, "r") as f:
            raw = json.load(f)
        self.samples = raw["samples"] if isinstance(raw, dict) and "samples" in raw else raw
        if not isinstance(self.samples, list):
            raise ValueError("Dataset annotation must be a list or contain a 'samples' list.")

        self.image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _choose_text(self, sample: Dict[str, Any]) -> str:
        names = [sample["canonical_name"], *sample.get("aliases", [])]
        names = [name for name in names if isinstance(name, str) and name.strip()]
        if not names:
            raise ValueError(f"Sample has no usable canonical_name or aliases: {sample}")
        if self.alias_mode == "canonical":
            return names[0]
        if self.alias_mode == "all":
            return random.choice(names)
        return random.choice(names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_path = resolve_path(sample["image"], self.image_dir)
        mask_path = resolve_path(sample["mask"], self.mask_dir)

        image = Image.open(image_path).convert("RGB")
        mask = np.load(mask_path).astype(np.float32)
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        mask = torch.from_numpy(mask)[None]
        mask = F.interpolate(
            mask[None], size=(self.resolution, self.resolution), mode="nearest"
        )[0]
        mask = (mask > 0.5).float()

        return {
            "image": self.image_transform(image),
            "mask": mask,
            "text": self._choose_text(sample),
            "image_path": image_path,
            "mask_path": mask_path,
        }


def collate_samples(samples: List[Dict[str, Any]]) -> Batch:
    return Batch(
        images=torch.stack([s["image"] for s in samples]),
        masks=torch.stack([s["mask"] for s in samples]),
        texts=[s["text"] for s in samples],
        image_paths=[s["image_path"] for s in samples],
        mask_paths=[s["mask_path"] for s in samples],
    )


class CrossAttentionFusion(nn.Module):
    def __init__(self, sam_dim: int = 256, text_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.sam_norm = nn.LayerNorm(sam_dim)
        self.text_norm = nn.LayerNorm(sam_dim)
        self.text_proj = nn.Linear(text_dim, sam_dim)
        self.cross_attn = nn.MultiheadAttention(sam_dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(sam_dim)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        sam_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_first_sam = sam_tokens.transpose(0, 1)
        projected_text = self.text_proj(text_tokens)
        key_padding_mask = None
        if text_attention_mask is not None:
            key_padding_mask = ~text_attention_mask.bool()
        fused, _ = self.cross_attn(
            query=self.sam_norm(batch_first_sam),
            key=self.text_norm(projected_text),
            value=projected_text,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = batch_first_sam + torch.tanh(self.gate) * fused
        return self.out_norm(fused).transpose(0, 1)


class TextConditionedSAM3LoRA(nn.Module):
    def __init__(self, sam3: SegmentationModel, config: Dict[str, Any]):
        super().__init__()
        self.sam3_wrapper = sam3
        self.sam3 = sam3.model
        self.config = config
        self.device_name = config.get("device", "cuda")
        self.resolution = config.get("data", {}).get("resolution", 1008)
        self.confidence_threshold = config.get("inference", {}).get(
            "confidence_threshold", 0.5
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
        for param in self.sam3.parameters():
            param.requires_grad = False

    def _require_peft(self) -> None:
        if get_peft_model is None:
            raise ImportError(
                "HuggingFace PEFT is required for LoRA. Install it with "
                "`pip install peft` in this environment."
            )

    def _apply_lora(self, cfg: Dict[str, Any]) -> None:
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
        if self.sam3.segmentation_head is not None:
            self.sam3.segmentation_head = get_peft_model(
                self.sam3.segmentation_head, lora_cfg
            )

    def _build_text_encoder(
        self, cfg: Dict[str, Any]
    ) -> Tuple[AutoTokenizer, nn.Module, int]:
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
        return [p for p in self.parameters() if p.requires_grad]

    def _external_text_tokens(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def predict_mask(self, image: Image.Image, text: str, device: torch.device) -> torch.Tensor:
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
        logits = select_text_conditioned_masks(out)[0]
        logits = interpolate(
            logits[None, None], (height, width), mode="bilinear", align_corners=False
        )[0, 0]
        return (logits.sigmoid() > self.confidence_threshold).detach().cpu()

    def save_checkpoint(self, output_dir: str, epoch: int, metrics: Dict[str, float]) -> None:
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.fusion.load_state_dict(checkpoint["fusion"], strict=strict)
        return checkpoint

    def load_lora(self, lora_dir: str) -> None:
        if PeftModel is None:
            self._require_peft()
        base = Path(lora_dir)
        self.sam3.backbone.vision_backbone = PeftModel.from_pretrained(
            self.sam3.backbone.vision_backbone, base / "image_encoder"
        )
        self.sam3.transformer.decoder = PeftModel.from_pretrained(
            self.sam3.transformer.decoder, base / "mask_decoder"
        )
        seg_head_dir = base / "segmentation_head"
        if seg_head_dir.exists() and self.sam3.segmentation_head is not None:
            self.sam3.segmentation_head = PeftModel.from_pretrained(
                self.sam3.segmentation_head, seg_head_dir
            )

    def export_lora(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.sam3.backbone.vision_backbone, "save_pretrained"):
            self.sam3.backbone.vision_backbone.save_pretrained(output_dir / "image_encoder")
        if hasattr(self.sam3.transformer.decoder, "save_pretrained"):
            self.sam3.transformer.decoder.save_pretrained(output_dir / "mask_decoder")
        if self.sam3.segmentation_head is not None and hasattr(
            self.sam3.segmentation_head, "save_pretrained"
        ):
            self.sam3.segmentation_head.save_pretrained(output_dir / "segmentation_head")


def select_text_conditioned_masks(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    logits = out["pred_logits"].squeeze(-1)
    best_idx = logits.argmax(dim=1)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    return out["pred_masks"][batch_idx, best_idx]


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.sigmoid()
    targets = targets.squeeze(1)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    targets = targets.squeeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = logits.sigmoid()
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def resize_targets(targets: torch.Tensor, pred_hw: Tuple[int, int]) -> torch.Tensor:
    if targets.shape[-2:] == pred_hw:
        return targets
    return F.interpolate(targets, size=pred_hw, mode="nearest")


def compute_loss(
    out: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_logits = select_text_conditioned_masks(out)
    targets = resize_targets(targets, pred_logits.shape[-2:])
    d_loss = dice_loss(pred_logits, targets)
    f_loss = focal_loss(pred_logits, targets)
    total = dice_weight * d_loss + focal_weight * f_loss
    return total, {"dice_loss": float(d_loss.detach()), "focal_loss": float(f_loss.detach())}


@torch.no_grad()
def compute_metrics(out: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
    logits = select_text_conditioned_masks(out)
    targets = resize_targets(targets, logits.shape[-2:]).squeeze(1).bool()
    preds = logits.sigmoid() > 0.5
    dims = tuple(range(1, preds.ndim))
    intersection = (preds & targets).sum(dim=dims).float()
    union = (preds | targets).sum(dim=dims).float()
    pred_sum = preds.sum(dim=dims).float()
    target_sum = targets.sum(dim=dims).float()
    iou = (intersection / union.clamp_min(1.0)).mean()
    dice = ((2.0 * intersection) / (pred_sum + target_sum).clamp_min(1.0)).mean()
    return {"miou": float(iou), "dice": float(dice)}


def move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        images=batch.images.to(device, non_blocking=True),
        masks=batch.masks.to(device, non_blocking=True),
        texts=batch.texts,
        image_paths=batch.image_paths,
        mask_paths=batch.mask_paths,
    )


def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    data_cfg = config["dataset"]
    train_cfg = config.get("training", {})
    dataset = DentalInstrumentDataset(
        annotation_path=data_cfg["annotation_path"],
        image_dir=data_cfg["input_img_dir"],
        mask_dir=data_cfg["mask_dir"],
        resolution=config.get("data", {}).get("resolution", 1008),
        alias_mode=train_cfg.get("alias_mode", "random"),
    )
    val_fraction = train_cfg.get("val_fraction", 0.2)
    val_size = max(1, int(round(len(dataset) * val_fraction))) if len(dataset) > 1 else 1
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        train_size, val_size = len(dataset), 0
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    if val_size > 0:
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_ds, val_ds = dataset, dataset
    batch_size = train_cfg.get("batch_size", 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=True,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=True,
        collate_fn=collate_samples,
    )
    return train_loader, val_loader


def build_model(config: Dict[str, Any], device: torch.device) -> TextConditionedSAM3LoRA:
    if device.type != "cuda":
        raise RuntimeError(
            "This SAM3 checkout allocates CUDA tensors during model construction. "
            "Use a CUDA device for training/inference, or update the SAM3 builder "
            "before CPU-only execution."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests CUDA, but torch.cuda.is_available() is false. This SAM3 "
            "checkout also allocates CUDA tensors during model construction, so run on "
            "a CUDA machine or update the SAM3 builder before CPU-only training."
        )
    sam3_config = dict(config["segmentation_model"])
    sam3_config.setdefault("prompt", config.get("inference", {}).get("default_prompt", "object"))
    sam3 = SegmentationModel(sam3_config, device=str(device))
    model = TextConditionedSAM3LoRA(sam3, config).to(device)
    return model


def train(config: Dict[str, Any]) -> None:
    seed_everything(config.get("seed", 42))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config, device)
    train_cfg = config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=train_cfg.get("mixed_precision", True) and device.type == "cuda"
    )
    use_autocast = autocast_enabled(device, train_cfg.get("mixed_precision", True))
    amp_dtype = autocast_dtype(device, train_cfg.get("amp_dtype", "float16"))
    output_dir = train_cfg.get("output_dir", "./outputs/sam3_lora")
    best_miou = -1.0

    for epoch in range(1, train_cfg.get("epochs", 10) + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                out = model(batch.images, batch.texts)
                loss, loss_parts = compute_loss(
                    out,
                    batch.masks,
                    dice_weight=train_cfg.get("dice_weight", 1.0),
                    focal_weight=train_cfg.get("focal_weight", 1.0),
                )
            scaler.scale(loss).backward()
            if train_cfg.get("grad_clip_norm"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.trainable_parameters(), train_cfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            if step % train_cfg.get("log_every", 10) == 0:
                print(
                    f"epoch={epoch} step={step} loss={running_loss / step:.4f} "
                    f"dice_loss={loss_parts['dice_loss']:.4f} focal_loss={loss_parts['focal_loss']:.4f}"
                )

        metrics = validate(model, val_loader, device, amp_dtype, use_autocast)
        print(
            f"epoch={epoch} train_loss={running_loss / max(len(train_loader), 1):.4f} "
            f"val_miou={metrics['miou']:.4f} val_dice={metrics['dice']:.4f}"
        )
        model.save_checkpoint(output_dir, epoch, metrics)
        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            model.save_checkpoint(Path(output_dir) / "best", epoch, metrics)


@torch.no_grad()
def validate(
    model: TextConditionedSAM3LoRA,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    totals = {"miou": 0.0, "dice": 0.0}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(batch.images, batch.texts)
        metrics = compute_metrics(out, batch.masks)
        totals["miou"] += metrics["miou"]
        totals["dice"] += metrics["dice"]
        count += 1
    return {k: v / max(count, 1) for k, v in totals.items()}


def run_predict(args: argparse.Namespace) -> None:
    config = get_config(args.config)
    device = torch.device(args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config, device)
    checkpoint_path = args.checkpoint or config.get("inference", {}).get("checkpoint")
    lora_dir = args.lora_dir or config.get("inference", {}).get("lora_dir")
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_checkpoint(checkpoint_path, strict=False)
    elif checkpoint_path:
        print(f"Checkpoint not found, using current weights: {checkpoint_path}")
    if lora_dir and Path(lora_dir).exists():
        model.load_lora(lora_dir)
    elif lora_dir:
        print(f"LoRA directory not found, using current adapters: {lora_dir}")
    model.eval()

    image = Image.open(args.image).convert("RGB")
    mask = model.predict_mask(image, args.object, device)
    output_path = args.output or "prediction_mask.npy"
    np.save(output_path, mask.numpy().astype(np.uint8))
    print(f"Saved mask to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run SAM3 LoRA text segmentation.")
    parser.add_argument("--config", default="./configs/config.yaml")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train LoRA adapters and fusion module.")

    predict_parser = subparsers.add_parser("predict", help="Predict one object mask.")
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--object", required=True)
    predict_parser.add_argument("--output", default="prediction_mask.npy")
    predict_parser.add_argument("--checkpoint")
    predict_parser.add_argument("--lora-dir")
    predict_parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in (None, "train"):
        train(get_config(args.config))
    elif args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
