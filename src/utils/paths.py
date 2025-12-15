import os
from pathlib import Path
from typing import Union

import yaml


PathLike = Union[str, Path]


def get_project_root() -> Path:
    """
    Returns the project root directory assuming this file lives in src/utils/.
    """
    return Path(__file__).resolve().parents[2]


def load_config(config_path: PathLike) -> dict:
    """
    Load a YAML configuration file.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = get_project_root() / config_path
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: PathLike) -> Path:
    """
    Resolve a path relative to the project root.
    """
    rel = Path(relative_path)
    if rel.is_absolute():
        return rel
    return get_project_root() / rel


def ensure_dir(path: PathLike) -> Path:
    """
    Ensure a directory exists and return it.
    """
    path = resolve_path(path)
    os.makedirs(path, exist_ok=True)
    return path


def get_data_paths(cfg: dict) -> dict:
    """
    Convenience helper to get commonly used data paths from config.
    """
    paths_cfg = cfg.get("paths", {})
    return {
        "raw_data": resolve_path(paths_cfg.get("raw_data", "data/raw/colorectal-cancer-wsi")),
        "processed_images": ensure_dir(paths_cfg.get("processed_images", "data/processed/images")),
        "processed_masks": ensure_dir(paths_cfg.get("processed_masks", "data/processed/masks")),
        "splits": ensure_dir(paths_cfg.get("splits", "data/splits")),
        "checkpoints": ensure_dir(paths_cfg.get("checkpoints", "checkpoints")),
        "logs": ensure_dir(paths_cfg.get("logs", "outputs/logs")),
        "visualizations": ensure_dir(paths_cfg.get("visualizations", "outputs/visualizations")),
    }



