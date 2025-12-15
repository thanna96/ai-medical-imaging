from pathlib import Path
from typing import Iterable, Optional

import csv
from datetime import datetime

from .paths import ensure_dir


def create_csv_logger(log_dir: Path, filename: str = "training_log.csv") -> Path:
    log_dir = ensure_dir(log_dir)
    log_path = log_dir / filename
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "epoch", "phase", "loss", "metric"])
    return log_path


def append_log_row(
    log_path: Path,
    epoch: int,
    phase: str,
    loss: float,
    metric: Optional[float] = None,
) -> None:
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [datetime.utcnow().isoformat(), epoch, phase, float(loss), "" if metric is None else float(metric)]
        )


def format_metrics(metrics: dict) -> str:
    parts: Iterable[str] = (f"{k}: {v:.4f}" for k, v in metrics.items())
    return " | ".join(parts)



