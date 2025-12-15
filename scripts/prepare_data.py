import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.paths import load_config, get_data_paths, ensure_dir


def find_image_mask_pairs(raw_root: Path) -> List[Tuple[Path, Path]]:
    """
    Discover image/mask pairs in the raw dataset directory.

    This implementation assumes that images are stored in a directory that
    contains 'Images' or 'image' in its path and masks in a directory that
    contains 'Mask' or 'mask'. It then matches files by stem (filename without extension).
    You may need to adjust this logic depending on the exact Kaggle layout.
    """
    image_files: List[Path] = []
    mask_files: List[Path] = []

    for p in raw_root.rglob("*"):
        if not p.is_file():
            continue
        lower = p.name.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            if "mask" in p.as_posix().lower():
                mask_files.append(p)
            else:
                image_files.append(p)

    mask_map = {p.stem: p for p in mask_files}
    pairs: List[Tuple[Path, Path]] = []
    for img in image_files:
        m = mask_map.get(img.stem)
        if m is not None:
            pairs.append((img, m))

    if not pairs:
        raise RuntimeError(
            f"No image/mask pairs found under {raw_root}. "
            "Please check the dataset structure and adjust `find_image_mask_pairs` accordingly."
        )

    return pairs


def resize_image_and_mask(
    image_path: Path,
    mask_path: Path,
    output_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    img_resized = cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    # Use nearest neighbor for masks to preserve class indices
    mask_resized = cv2.resize(mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

    # If mask has 3 channels, convert to single-channel by taking the first channel
    if mask_resized.ndim == 3:
        mask_resized = mask_resized[:, :, 0]

    return img_resized, mask_resized


def save_pair(
    img: np.ndarray,
    mask: np.ndarray,
    img_out: Path,
    mask_out: Path,
) -> None:
    img_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    # Save image as RGB PNG
    cv2.imwrite(str(img_out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # Save mask as single-channel PNG
    cv2.imwrite(str(mask_out), mask)


def create_splits(ids: List[str], splits_dir: Path, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    train_ids, temp_ids = train_test_split(ids, train_size=train_ratio, random_state=seed, shuffle=True)
    relative_val = val_ratio / (1.0 - train_ratio)
    val_ids, test_ids = train_test_split(temp_ids, test_size=1.0 - relative_val, random_state=seed, shuffle=True)

    splits_dir.mkdir(parents=True, exist_ok=True)

    (splits_dir / "train.txt").write_text("\n\n".join(sorted(train_ids)))
    (splits_dir / "val.txt").write_text("\n\n".join(sorted(val_ids)))
    (splits_dir / "test.txt").write_text("\n\n".join(sorted(test_ids)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EBHI-Seg / Colorectal WSI dataset.")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_paths = get_data_paths(cfg)

    raw_root: Path = data_paths["raw_data"]
    images_root: Path = data_paths["processed_images"]
    masks_root: Path = data_paths["processed_masks"]
    splits_root: Path = data_paths["splits"]

    ensure_dir(images_root)
    ensure_dir(masks_root)
    ensure_dir(splits_root)

    image_size: int = int(cfg.get("data", {}).get("image_size", 256))

    pairs = find_image_mask_pairs(raw_root)
    ids: List[str] = []

    for img_path, mask_path in pairs:
        sample_id = img_path.stem
        ids.append(sample_id)
        resized_img, resized_mask = resize_image_and_mask(img_path, mask_path, image_size)

        img_out = images_root / f"{sample_id}.png"
        mask_out = masks_root / f"{sample_id}.png"
        save_pair(resized_img, resized_mask, img_out, mask_out)

    create_splits(ids, splits_root)
    print(f"Prepared {len(ids)} samples.")
    print(f"Images saved to: {images_root}")
    print(f"Masks saved to: {masks_root}")
    print(f"Splits saved to: {splits_root}")


if __name__ == "__main__":
    main()


