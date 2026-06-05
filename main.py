"""Train and run text-conditioned SAM3 segmentation with PEFT LoRA adapters."""

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, random_split
from models import TextConditionedSAM3LoRA, SAM3Wrapper
from .dataset import DentalInstrumentDataset
from .utils import (
    get_config, 
    seed_everything,
    autocast_dtype,
    autocast_enabled, 
    collate_samples,
    move_batch, 
    compute_loss, 
    compute_metrics
)


def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from YAML configuration."""

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
    """Build the SAM3 LoRA training model on a CUDA device."""

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
    sam3 = SAM3Wrapper(sam3_config, device=str(device))
    model = TextConditionedSAM3LoRA(sam3, config).to(device)
    return model


def train(config: Dict[str, Any]) -> None:
    """Train LoRA adapters and the cross-attention fusion module."""

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
    scaler = torch.amp.GradScaler(
        "cuda", enabled=train_cfg.get("mixed_precision", True) and device.type == "cuda"
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
    """Evaluate a model over a validation loader and average metrics."""

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
    """Run single-image text-conditioned segmentation from parsed CLI args."""

    config = get_config(args.config)
    device = torch.device(args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = args.checkpoint or config.get("inference", {}).get("checkpoint")
    lora_dir = args.lora_dir or config.get("inference", {}).get("lora_dir")

    model = build_model(config, device)
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
    """Parse the training and inference command-line interface."""

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
