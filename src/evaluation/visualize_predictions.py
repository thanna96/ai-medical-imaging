import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.ebhi_seg_dataset import EBHISegDataset
from src.datasets.transforms import get_test_transforms
from src.models.unet import UNet
from src.utils.paths import load_config, get_data_paths, resolve_path, ensure_dir
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize U-Net predictions on colorectal WSI data.")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Split to visualize.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to visualize.")
    return parser.parse_args()


def create_color_map(num_classes: int) -> np.ndarray:
    base_colors = np.array(
        [
            [0, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
            [255, 0, 0],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
        ],
        dtype=np.uint8,
    )
    if num_classes <= base_colors.shape[0]:
        return base_colors[:num_classes]
    reps = int(np.ceil(num_classes / base_colors.shape[0]))
    return np.vstack([base_colors] * reps)[:num_classes]


def mask_to_color(mask: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(color_map.shape[0]):
        color_mask[mask == cls_idx] = color_map[cls_idx]
    return color_mask


def overlay(image: np.ndarray, mask_color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})

    data_paths = get_data_paths(cfg)

    image_size = int(data_cfg.get("image_size", 256))
    num_classes = int(data_cfg.get("num_classes", model_cfg.get("num_classes", 6)))
    norm_cfg = data_cfg.get("normalization", {})
    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])
    ignore_index = int(loss_cfg.get("ignore_index", -1))

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device_str = train_cfg.get("device", "cpu")
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")

    split_file = "val.txt" if args.split == "val" else "test.txt"
    split_path = resolve_path(data_paths["splits"] / split_file)
    transforms = get_test_transforms(image_size, mean, std)

    dataset = EBHISegDataset(
        split_file=str(split_path),
        images_dir=str(data_paths["processed_images"]),
        masks_dir=str(data_paths["processed_masks"]),
        transform=transforms,
        ignore_index=ignore_index,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = UNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=num_classes,
        base_channels=int(model_cfg.get("base_channels", 32)),
    ).to(device)

    ckpt_path = resolve_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = ensure_dir(data_paths["visualizations"])
    color_map = create_color_map(num_classes)

    count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            ids = batch["id"]

            logits = model(images)
            preds = logits.softmax(dim=1).argmax(dim=1)

            for img_t, gt_t, pred_t, sample_id in zip(images, masks, preds, ids):
                img = (img_t.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                gt = gt_t.cpu().numpy().astype(np.int64)
                pred = pred_t.cpu().numpy().astype(np.int64)

                gt_color = mask_to_color(gt, color_map)
                pred_color = mask_to_color(pred, color_map)

                gt_overlay = overlay(img, gt_color, alpha=0.5)
                pred_overlay = overlay(img, pred_color, alpha=0.5)

                base = sample_id
                cv2.imwrite(str(out_dir / f"{base}_image.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out_dir / f"{base}_gt.png"), cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out_dir / f"{base}_pred.png"), cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))

                count += 1
                if count >= args.num_samples:
                    print(f"Saved {count} visualizations to {out_dir}")
                    return

    print(f"Saved {count} visualizations to {out_dir}")


if __name__ == "__main__":
    main()


