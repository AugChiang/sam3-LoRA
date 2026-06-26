import json
import torch
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from typing import Dict, Any, Tuple

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
        augmentation: Dict[str, Any] = None,
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
            augmentation: Optional training augmentation configuration. Geometry is
                applied to image and mask; color jitter is applied to image only.
        """

        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.resolution = resolution
        self.alias_mode = alias_mode
        self.augmentation = augmentation or {}
        self.use_augmentation = self.augmentation.get("enabled", False)

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

    def _apply_augmentation(self, image: Image.Image, mask: torch.Tensor) -> Tuple[Image.Image, torch.Tensor]:
        """Apply paired geometry to image/mask and image-only color jitter."""

        cfg = self.augmentation
        if random.random() < cfg.get("horizontal_flip_prob", 0.0):
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < cfg.get("vertical_flip_prob", 0.0):
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        degrees = cfg.get("rotation_degrees", 0.0)
        translate_frac = cfg.get("translate", 0.0)
        scale_min = cfg.get("scale_min", 1.0)
        scale_max = cfg.get("scale_max", 1.0)
        if degrees or translate_frac or scale_min != 1.0 or scale_max != 1.0:
            angle = random.uniform(-degrees, degrees)
            max_dx = int(round(image.width * translate_frac))
            max_dy = int(round(image.height * translate_frac))
            translate = (
                random.randint(-max_dx, max_dx) if max_dx > 0 else 0,
                random.randint(-max_dy, max_dy) if max_dy > 0 else 0,
            )
            scale = random.uniform(scale_min, scale_max)
            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0.0,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0.0,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        image = self._jitter_image(image, "brightness", TF.adjust_brightness)
        image = self._jitter_image(image, "contrast", TF.adjust_contrast)
        image = self._jitter_image(image, "saturation", TF.adjust_saturation)
        return image, mask

    def _jitter_image(self, image: Image.Image, key: str, adjust_fn) -> Image.Image:
        jitter = self.augmentation.get(key, 0.0)
        if not jitter:
            return image
        factor = random.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)
        return adjust_fn(image, factor)

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
        if self.use_augmentation:
            image, mask = self._apply_augmentation(image, mask)
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
