import torch
from torch.utils.data import DataLoader, Subset
from models import TextConditionedSAM3LoRA, SAM3Wrapper

try:
    from .dataset import CustomDataset
    from .utils import *
except ImportError:
    from dataset import CustomDataset
    from utils import *


def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from YAML configuration."""

    data_cfg = config["dataset"]
    train_cfg = config.get("training", {})
    dataset_args = {
        "annotation_path": data_cfg["annotation_path"],
        "image_dir": data_cfg["input_img_dir"],
        "mask_dir": data_cfg["mask_dir"],
        "resolution": config.get("data", {}).get("resolution", 1008),
        "alias_mode": train_cfg.get("alias_mode", "random"),
    }
    train_dataset = CustomDataset(
        **dataset_args,
        augmentation=train_cfg.get("augmentation", {}),
    )
    val_dataset = CustomDataset(
        **dataset_args,
        augmentation={"enabled": False},
    )
    dataset_size = len(train_dataset)
    val_fraction = train_cfg.get("val_fraction", 0.2)
    val_size = max(1, int(round(dataset_size * val_fraction))) if dataset_size > 1 else 1
    train_size = max(1, dataset_size - val_size)
    if train_size + val_size > dataset_size:
        train_size, val_size = dataset_size, 0
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    if val_size > 0:
        train_ds = Subset(train_dataset, train_indices)
        val_ds = Subset(val_dataset, val_indices)
    else:
        train_ds = train_dataset
        val_ds = val_dataset
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


def build_sam3_wrapper(config: Dict[str, Any], device: torch.device) -> SAM3Wrapper:
    """Build the original SAM3 wrapper without LoRA or fusion modules."""

    require_cuda_device(device)
    sam3_config = dict(config["segmentation_model"])
    sam3_config.setdefault("prompt", config.get("inference", {}).get("default_prompt", "object"))
    return SAM3Wrapper(sam3_config, device=str(device))


def build_model(config: Dict[str, Any], device: torch.device) -> TextConditionedSAM3LoRA:
    """Build the SAM3 LoRA training model on a CUDA device."""

    sam3 = build_sam3_wrapper(config, device)
    model = TextConditionedSAM3LoRA(sam3, config).to(device)
    return model