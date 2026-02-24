"""
Global configuration loader.

Reads ``config.yaml`` located at the repository root and exposes a ``CFG``
singleton available to every module via::

    from src.config import CFG

    lr = CFG["training"]["lr"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_CONFIG_PATH: Path = _REPO_ROOT / "config.yaml"


def load_config(path: Path | str = _CONFIG_PATH) -> dict[str, Any]:
    """Load and return the YAML configuration file as a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


#: Module-level singleton — loaded once at import time.
CFG: dict[str, Any] = load_config()
