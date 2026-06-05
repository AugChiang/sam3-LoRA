import os
import yaml
import numpy as np
import torch
import random
import torch.nn.functional as F

from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class Batch:
    """Mini-batch produced by the dental instrument dataset collator."""
    images: torch.Tensor
    masks: torch.Tensor
    texts: List[str]
    image_paths: List[str]
    mask_paths: List[str]

def get_config(config_path=None) -> dict:
    if config_path is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(this_file_dir))
        config_path = os.path.join(repo_root, 'configs', 'config.yaml')
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


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
    """Resolve a possibly relative dataset path against a base directory."""

    p = Path(path)
    return str(p if p.is_absolute() else Path(base_dir) / p)

def collate_samples(samples: List[Dict[str, Any]]) -> Batch:
    """Stack transformed dataset samples into a training batch."""

    return Batch(
        images=torch.stack([s["image"] for s in samples]),
        masks=torch.stack([s["mask"] for s in samples]),
        texts=[s["text"] for s in samples],
        image_paths=[s["image_path"] for s in samples],
        mask_paths=[s["mask_path"] for s in samples],
    )


def require_cuda_device(device: torch.device) -> None:
    """Validate that this SAM3 checkout can construct CUDA-only model components."""
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


# losses
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute soft Dice loss from mask logits and binary targets."""

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
    """Compute sigmoid focal loss for foreground/background mask pixels."""

    targets = targets.squeeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = logits.sigmoid()
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()

def resize_targets(targets: torch.Tensor, pred_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize target masks to match prediction spatial dimensions."""

    if targets.shape[-2:] == pred_hw:
        return targets
    return F.interpolate(targets, size=pred_hw, mode="nearest")


def select_text_conditioned_masks(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Select the highest-scoring predicted mask for each batch item."""

    logits = out["pred_logits"].squeeze(-1)
    best_idx = logits.argmax(dim=1)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    return out["pred_masks"][batch_idx, best_idx]

def compute_loss(
    out: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute weighted Dice plus Focal training loss."""

    pred_logits = select_text_conditioned_masks(out)
    targets = resize_targets(targets, pred_logits.shape[-2:])
    d_loss = dice_loss(pred_logits, targets)
    f_loss = focal_loss(pred_logits, targets)
    total = dice_weight * d_loss + focal_weight * f_loss
    return total, {"dice_loss": float(d_loss.detach()), "focal_loss": float(f_loss.detach())}


@torch.no_grad()
def compute_metrics(out: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
    """Compute validation mIoU and Dice from SAM3 mask outputs."""

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
    """Move image and mask tensors to the target device while preserving metadata."""

    return Batch(
        images=batch.images.to(device, non_blocking=True),
        masks=batch.masks.to(device, non_blocking=True),
        texts=batch.texts,
        image_paths=batch.image_paths,
        mask_paths=batch.mask_paths,
    )


def select_original_sam3_mask(output: Dict[str, Any], image_size: Tuple[int, int]) -> np.ndarray:
    """Select the best mask from a SAM3Processor output state."""

    width, height = image_size
    masks = output.get("masks")
    if masks is None or len(masks) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(masks, torch.Tensor):
        masks_tensor = masks.detach().cpu()
    else:
        masks_tensor = torch.as_tensor(masks)

    scores = output.get("scores")
    if scores is not None and len(scores) > 0:
        scores_tensor = scores.detach().cpu() if isinstance(scores, torch.Tensor) else torch.as_tensor(scores)
        mask_idx = int(scores_tensor.argmax().item())
    else:
        mask_idx = 0

    mask = masks_tensor[mask_idx]
    while mask.ndim > 2:
        mask = mask.squeeze(0)
    return (mask.numpy() > 0).astype(np.uint8)