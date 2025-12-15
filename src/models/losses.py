from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Standard multi-class cross-entropy for segmentation.
    logits: (N, C, H, W)
    targets: (N, H, W) with class indices
    """
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


def _flatten_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns flattened probabilities and one-hot targets with ignored pixels removed.
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)

    # (N, C, H, W) -> (N*H*W, C)
    probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes)
    targets = targets.view(-1)

    if ignore_index >= 0:
        mask = targets != ignore_index
        probs = probs[mask]
        targets = targets[mask]

    # (N*H*W, C)
    one_hot = F.one_hot(targets, num_classes=num_classes).float()
    return probs, one_hot


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Soft per-class Dice loss averaged over classes.
    """
    probs, one_hot = _flatten_logits_and_targets(logits, targets, ignore_index)
    if one_hot.numel() == 0:
        # No valid pixels; return zero to avoid breaking training
        return logits.new_tensor(0.0)

    intersection = torch.sum(probs * one_hot, dim=0)
    cardinality = torch.sum(probs + one_hot, dim=0)
    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
    dice_loss_value = 1.0 - dice_per_class.mean()
    return dice_loss_value


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_weight: float = 1.0,
    ignore_index: int = -1,
) -> torch.Tensor:
    ce = cross_entropy_loss(logits, targets, ignore_index=ignore_index)
    d = dice_loss(logits, targets, ignore_index=ignore_index)
    return ce + dice_weight * d



