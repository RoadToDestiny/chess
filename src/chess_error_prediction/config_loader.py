"""
Central configuration loader for chess error prediction experiments.
Loads settings from config/config.yaml with fallback to research defaults.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def find_config_path() -> Path:
    """Finds the default config/config.yaml file relative to package or project root."""
    current = Path(__file__).resolve()
    for parent in [current.parents[1], current.parents[2], current.parents[3]]:
        candidate = parent / "config" / "config.yaml"
        if candidate.is_file():
            return candidate
    return Path("config/config.yaml")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads unified configuration from YAML file.
    Returns empty dict if file does not exist.
    """
    path = config_path or find_config_path()
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def get_model_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Extracts 'model' configuration section."""
    cfg = load_config(config_path)
    return cfg.get("model", {})


def get_stockfish_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Extracts 'stockfish' configuration section."""
    cfg = load_config(config_path)
    return cfg.get("stockfish", {})


def get_filter_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Extracts 'filter' configuration section."""
    cfg = load_config(config_path)
    return cfg.get("filter", {})
