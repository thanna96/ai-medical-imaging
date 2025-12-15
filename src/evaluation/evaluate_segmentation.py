import argparse

import torch
from torch.utils.data import DataLoader

from src.datasets.ebhi_seg_dataset import EBHISegDataset
from src.datasets.transforms import get_test_transforms
from src.evaluation.metrics import compute_confusion_matrix, metrics_from_confusion_matrix
from src.models.unet import UNet
from src.utils.paths import load_config, get_data_paths, resolve_path
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained U-Net on test split.")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint.")
    return parser.parse_args()


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

    test_split = resolve_path(data_paths["splits"] / "test.txt")
    test_transforms = get_test_transforms(image_size, mean, std)

    test_dataset = EBHISegDataset(
        split_file=str(test_split),
        images_dir=str(data_paths["processed_images"]),
        masks_dir=str(data_paths["processed_masks"]),
        transform=test_transforms,
        ignore_index=ignore_index,
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=num_classes,
        base_channels=int(model_cfg.get("base_channels", 32)),
    ).to(device)

    ckpt_path = resolve_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cm = torch.zeros((num_classes, num_classes), device=device, dtype=torch.long)

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            cm += compute_confusion_matrix(
                logits, masks, num_classes=num_classes, ignore_index=ignore_index
            )

    metrics = metrics_from_confusion_matrix(cm)
    print("Test results:")
    print(f"  Mean IoU:   {metrics['mean_iou'].item():.4f}")
    print(f"  Mean Dice:  {metrics['mean_dice'].item():.4f}")
    print(f"  Pixel Acc:  {metrics['pixel_acc'].item():.4f}")
    print("  Per-class IoU:", metrics["per_class_iou"].cpu().numpy())
    print("  Per-class Dice:", metrics["per_class_dice"].cpu().numpy())


if __name__ == "__main__":
    main()


