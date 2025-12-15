import argparse

import torch
from torch.utils.data import DataLoader

from src.datasets.ebhi_seg_dataset import EBHISegDataset
from src.datasets.transforms import get_train_transforms, get_val_transforms
from src.models.losses import combined_loss
from src.models.unet import UNet
from src.training.engine import train_one_epoch, validate
from src.training.scheduler import create_scheduler
from src.utils.logging_utils import create_csv_logger, append_log_row, format_metrics
from src.utils.paths import load_config, get_data_paths, ensure_dir, resolve_path
from src.utils.seed import set_seed


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train U-Net for colorectal WSI segmentation.")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})
    scheduler_cfg = cfg.get("scheduler", {})

    data_paths = get_data_paths(cfg)

    image_size = int(data_cfg.get("image_size", 256))
    num_classes = int(data_cfg.get("num_classes", model_cfg.get("num_classes", 6)))
    norm_cfg = data_cfg.get("normalization", {})
    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])

    batch_size = args.batch_size or int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 2))
    num_epochs = args.epochs or int(train_cfg.get("num_epochs", 30))
    lr = float(train_cfg.get("learning_rate", 1.0e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-5))
    grad_clip = float(train_cfg.get("gradient_clip", 0.0))
    seed = int(train_cfg.get("seed", 42))
    device_str = train_cfg.get("device", "cpu")
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")

    ignore_index = int(loss_cfg.get("ignore_index", -1))
    dice_weight = float(loss_cfg.get("dice_weight", 1.0))

    set_seed(seed)

    # Datasets & loaders
    train_split = resolve_path(data_paths["splits"] / "train.txt")
    val_split = resolve_path(data_paths["splits"] / "val.txt")

    train_transforms = get_train_transforms(image_size, mean, std)
    val_transforms = get_val_transforms(image_size, mean, std)

    train_dataset = EBHISegDataset(
        split_file=str(train_split),
        images_dir=str(data_paths["processed_images"]),
        masks_dir=str(data_paths["processed_masks"]),
        transform=train_transforms,
        ignore_index=ignore_index,
    )
    val_dataset = EBHISegDataset(
        split_file=str(val_split),
        images_dir=str(data_paths["processed_images"]),
        masks_dir=str(data_paths["processed_masks"]),
        transform=val_transforms,
        ignore_index=ignore_index,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Model, optimizer, scheduler
    model = UNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=num_classes,
        base_channels=int(model_cfg.get("base_channels", 32)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = create_scheduler(optimizer, scheduler_cfg)

    def loss_fn(logits, targets):
        return combined_loss(
            logits,
            targets,
            dice_weight=dice_weight,
            ignore_index=ignore_index,
        )

    checkpoints_dir = ensure_dir(data_paths["checkpoints"])
    best_model_path = checkpoints_dir / "best_model.pt"

    logs_dir = ensure_dir(data_paths["logs"])
    log_path = create_csv_logger(logs_dir)

    best_val_metric = -1.0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            epoch=epoch,
        )
        append_log_row(
            log_path,
            epoch=epoch,
            phase="train",
            loss=train_metrics["loss"],
            metric=train_metrics["mean_dice"],
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            epoch=epoch,
            phase="val",
        )
        append_log_row(
            log_path,
            epoch=epoch,
            phase="val",
            loss=val_metrics["loss"],
            metric=val_metrics["mean_dice"],
        )

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train {format_metrics(train_metrics)} | "
            f"Val {format_metrics(val_metrics)}"
        )

        current_metric = val_metrics["mean_dice"]
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                best_model_path,
            )
            print(f"  -> New best model saved to {best_model_path} (mean_dice={best_val_metric:.4f})")

        if scheduler is not None:
            if scheduler.__class__.__name__.lower() == "reducelronplateau":
                scheduler.step(current_metric)
            else:
                scheduler.step()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    print("Training complete.")
    print(f"Best validation mean Dice: {best_val_metric:.4f}")


if __name__ == "__main__":
    main()


