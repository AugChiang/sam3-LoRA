"""Train and run text-conditioned SAM3 segmentation with PEFT LoRA adapters."""

import argparse
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from pathlib import Path
from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader

from models import TextConditionedSAM3LoRA, EarlyStopper
try:
    from .builder import *
    from .utils import *
except ImportError:
    from builder import *
    from utils import *


def train(config: Dict[str, Any]) -> None:
    """Train LoRA adapters and the cross-attention fusion module."""

    seed_everything(config.get("seed", 42))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config, device)
    train_cfg: dict = config.get("training", {})
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
    early_stop_cfg: dict = train_cfg.get("early_stop", {})
    early_stopper = None
    early_stop_monitor = early_stop_cfg.get("monitor", "miou")
    if early_stop_cfg.get("enabled", False):
        early_stopper = EarlyStopper(
            patience=early_stop_cfg.get("patience", 3),
            min_delta=early_stop_cfg.get("min_delta", 0.0),
            mode=early_stop_cfg.get("mode", "max"),
        )
    loss_histroy = {"dice_loss": [], "focal_loss": []}
    for epoch in range(1, train_cfg.get("epochs", 10) + 1):
        model.train()
        tr_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch: Batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                out = model(batch.images, batch.texts)
                loss, loss_dict = compute_loss(
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
            tr_loss += float(loss.detach())
            if step % train_cfg.get("log_every", 10) == 0:
                print(
                    f"epoch={epoch} step={step} loss={tr_loss / step:.4f} "
                    f"dice_loss={loss_dict['dice_loss']:.4f} focal_loss={loss_dict['focal_loss']:.4f}"
                )
            loss_histroy["dice_loss"].append(loss_dict['dice_loss'])
            loss_histroy["focal_loss"].append(loss_dict['focal_loss'])

        metrics = validate(model, val_loader, device, amp_dtype, use_autocast)
        print(
            f"epoch={epoch} train_loss={tr_loss / max(len(train_loader), 1):.4f} "
            f"val_miou={metrics['miou']:.4f} val_dice={metrics['dice']:.4f}"
        )
        model.save_checkpoint(output_dir, epoch, metrics)
        # save losses
        np.save(f"{output_dir}/dice-loss.npy", np.array(loss_histroy['dice_loss']))
        np.save(f"{output_dir}/focal-loss.npy", np.array(loss_histroy['dice_loss']))
        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            model.save_checkpoint(Path(output_dir) / "best", epoch, metrics)
        if early_stopper is not None:
            if early_stop_monitor not in metrics:
                raise KeyError(
                    f"early stopping monitor '{early_stop_monitor}' is not in validation metrics: "
                    f"{sorted(metrics.keys())}"
                )
            monitor_value = metrics[early_stop_monitor]
            if early_stopper.step(monitor_value):
                print(
                    f"early stopping at epoch={epoch}: "
                    f"{early_stop_monitor}={monitor_value:.4f}, "
                    f"best={early_stopper.best_score:.4f}, "
                    f"patience={early_stopper.patience}"
                )
                break


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
    if args.mode == "sam3":
        run_original_sam3_predict(args, config, device)
        return

    checkpoint_path = args.checkpoint or config.get("inference", {}).get("checkpoint")
    lora_dir = args.lora_dir or config.get("inference", {}).get("lora_dir")

    model: TextConditionedSAM3LoRA = build_model(config, device)
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
    mask = combine_mask(mask)
    output_path = args.output or "prediction_mask.npy"
    np.save(output_path, mask.astype(np.uint8))
    print(f"Saved mask to {output_path}")


def run_original_sam3_predict(
    args: argparse.Namespace,
    config: Dict[str, Any],
    device: torch.device,
) -> None:
    """Run base SAM3 text-prompt inference without LoRA or fusion modules."""

    sam3 = build_sam3_wrapper(config, device)
    image = Image.open(args.image).convert("RGB")
    output = sam3.pred(image, txt_prompt=args.object)
    mask = select_original_sam3_mask(output, image.size)
    output_path = args.output or "prediction_mask.npy"
    np.save(output_path, mask.astype(np.uint8))
    print(f"Saved base SAM3 mask to {output_path}")


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
    predict_parser.add_argument(
        "--mode",
        choices=["lora", "sam3"],
        default="lora",
        help="Use trained LoRA adapters or original SAM3 checkpoint-only inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in (None, "train"):
        train(get_config(args.config))
    elif args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
