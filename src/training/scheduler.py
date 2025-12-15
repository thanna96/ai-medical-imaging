from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, _LRScheduler


def create_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: Optional[dict],
) -> Optional[_LRScheduler]:
    """
    Create a learning rate scheduler from config.

    Example cfg:
    {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-6
    }
    """
    if not scheduler_cfg:
        return None

    sch_type = scheduler_cfg.get("type", "").lower()

    if sch_type == "reduce_on_plateau":
        factor = scheduler_cfg.get("factor", 0.5)
        patience = scheduler_cfg.get("patience", 5)
        min_lr = scheduler_cfg.get("min_lr", 1e-6)
        return ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience, min_lr=min_lr)

    if sch_type == "step":
        step_size = scheduler_cfg.get("step_size", 10)
        gamma = scheduler_cfg.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    return None



