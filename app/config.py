"""
Shared runtime configuration and path constants for the Axis Descriptor Lab.

This module centralises the filesystem paths and version metadata that were
previously computed directly inside ``app.main``.  Route modules and services
can import these constants without depending on the application entrypoint.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
LOGS_DIR = HERE.parent / "logs"
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemma2:2b")

_PYPROJECT = HERE.parent / "pyproject.toml"
with open(_PYPROJECT, "rb") as _f:
    APP_VERSION: str = tomllib.load(_f)["project"]["version"]
