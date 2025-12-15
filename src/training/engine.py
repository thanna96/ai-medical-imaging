import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.evaluation.metrics import compute_confusion_matrix, metrics_from_confusion_matrix


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    num_classes: int,
    ignore_index: int = -1,
    epoch: int = 0,
) -> dict:
    model.train()
    running_loss = 0.0
    cm = torch.zeros((num_classes, num_classes), device=device, dtype=torch.long)

    pbar = tqdm(dataloader, desc=f"Train {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        with torch.no_grad():
            cm += compute_confusion_matrix(
                logits, masks, num_classes=num_classes, ignore_index=ignore_index
            )

        if batch_idx % 10 == 0:
            metrics = metrics_from_confusion_matrix(cm)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{metrics['mean_dice'].item():.3f}",
                miou=f"{metrics['mean_iou'].item():.3f}",
            )

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = metrics_from_confusion_matrix(cm)
    metrics_float = {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in metrics.items()}
    metrics_float["loss"] = epoch_loss
    return metrics_float


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn,
    device: torch.device,
    num_classes: int,
    ignore_index: int = -1,
    epoch: int = 0,
    phase: str = "Val",
) -> dict:
    model.eval()
    running_loss = 0.0
    cm = torch.zeros((num_classes, num_classes), device=device, dtype=torch.long)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"{phase} {epoch}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = loss_fn(logits, masks)

            running_loss += loss.item() * images.size(0)
            cm += compute_confusion_matrix(
                logits, masks, num_classes=num_classes, ignore_index=ignore_index
            )

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = metrics_from_confusion_matrix(cm)
    metrics_float = {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in metrics.items()}
    metrics_float["loss"] = epoch_loss
    return metrics_float


