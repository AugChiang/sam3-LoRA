import json
import torch
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from typing import Dict, Any

try:
    from .utils import resolve_path
except ImportError:
    from utils import resolve_path


class CustomDataset(Dataset):
    """Load image, text alias, and binary mask samples for Out-of-Distribution objects (unseen objects)."""

    def __init__(
        self,
        annotation_path: str,
        image_dir: str,
        mask_dir: str,
        resolution: int = 1008,
        alias_mode: str = "random",
    ) -> None:
        """
        Initialize a dataset from JSON annotations and asset directories.

        Args:
            annotation_path: 
                JSON file containing either a sample list or 
                a dictionary with a top-level `samples` list.
            image_dir: Directory containing image files referenced by samples.
            mask_dir: Directory containing binary `.npy` mask files.
            resolution: Square training resolution used for image and mask resize.
            alias_mode: `canonical` always uses the canonical name; other values
                randomly sample from canonical name plus aliases.
        """

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
        """Return the number of annotated object-mask samples."""

        return len(self.samples)

    def _choose_text(self, sample: Dict[str, Any]) -> str:
        """
        Choose the text prompt for a sample, 
        including aliases when enabled.
        """

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
        """
        Returns:
            Dict[str, Any]:
                - image as [3, H, W]
                - mask as [1, H, W], 
                - selected text prompt
                - resolved source paths.
        """

        sample = self.samples[idx]
        image_path = resolve_path(sample["image"], self.image_dir)
        mask_path = resolve_path(sample["mask"], self.mask_dir)

        image = Image.open(image_path).convert("RGB")
        mask: np.ndarray = np.load(mask_path).astype(np.float32)
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
