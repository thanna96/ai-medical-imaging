from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.paths import resolve_path


class EBHISegDataset(Dataset):
    """
    PyTorch Dataset for EBHI-Seg / Colorectal WSI segmentation.

    Expects:
      - processed images in data/processed/images/<id>.png
      - processed masks in data/processed/masks/<id>.png
      - split files (train/val/test) listing IDs, one per line
    """

    def __init__(
        self,
        split_file: str,
        images_dir: str,
        masks_dir: str,
        transform: Optional[A.Compose] = None,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.images_dir = resolve_path(images_dir)
        self.masks_dir = resolve_path(masks_dir)
        self.transform = transform
        self.ignore_index = ignore_index

        split_path = resolve_path(split_file)
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        # Support blank lines
        ids: List[str] = []
        for line in split_path.read_text().splitlines():
            line = line.strip()
            if line:
                ids.append(line)

        if not ids:
            raise RuntimeError(f"No IDs found in split file: {split_path}")

        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, image_path: Path) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.astype(np.int64)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.ids[idx]
        img_path = self.images_dir / f"{sample_id}.png"
        mask_path = self.masks_dir / f"{sample_id}.png"

        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Default conversion to tensors if no transform is provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {"image": image, "mask": mask, "id": sample_id}



