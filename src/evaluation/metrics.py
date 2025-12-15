from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Compute confusion matrix for segmentation.
    Returns (C, C) where rows are ground truth and columns are predictions.
    """
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)  # (N, H, W)

        preds = preds.view(-1)
        targets = targets.view(-1)

        if ignore_index >= 0:
            mask = targets != ignore_index
            preds = preds[mask]
            targets = targets[mask]

        k = (targets >= 0) & (targets < num_classes)
        inds = num_classes * targets[k].to(torch.int64) + preds[k]
        cm = torch.bincount(inds, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    return cm


def metrics_from_confusion_matrix(cm: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute per-class and mean IoU/Dice and pixel accuracy from confusion matrix.
    """
    # True positives: diagonal
    tp = torch.diag(cm).float()
    # For IoU: denominator = TP + FP + FN
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    denom_iou = tp + fp + fn
    iou = torch.where(denom_iou > 0, tp / denom_iou.clamp(min=1.0), torch.zeros_like(tp))
    miou = iou.mean()

    # Dice = 2TP / (2TP + FP + FN)
    denom_dice = 2 * tp + fp + fn
    dice = torch.where(denom_dice > 0, 2 * tp / denom_dice.clamp(min=1.0), torch.zeros_like(tp))
    mdice = dice.mean()

    # Pixel accuracy
    total = cm.sum().float().clamp(min=1.0)
    acc = tp.sum() / total

    return {
        "per_class_iou": iou,
        "mean_iou": miou,
        "per_class_dice": dice,
        "mean_dice": mdice,
        "pixel_acc": acc,
    }


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> Dict[str, torch.Tensor]:
    cm = compute_confusion_matrix(logits, targets, num_classes=num_classes, ignore_index=ignore_index)
    return metrics_from_confusion_matrix(cm)



