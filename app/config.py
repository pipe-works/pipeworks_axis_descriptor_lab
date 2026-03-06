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
WORLD_ROOT = HERE / "worlds"
LAB_ONLY_ROOT = HERE / "lab_only"
LEGACY_PROMPTS_DIR = HERE / "prompts"
LEGACY_EXAMPLES_DIR = HERE / "examples"
LEGACY_LEXICONS_DIR = HERE / "data"
LEGACY_POLICY_BUNDLES_DIR = HERE / "artifacts" / "policy_bundles"
DEFAULT_WORLD_ID: str = os.getenv("LAB_DEFAULT_WORLD_ID", "pipeworks_web")
LEGACY_ROOTS: dict[str, Path] = {
    "prompts": LEGACY_PROMPTS_DIR,
    "examples": LEGACY_EXAMPLES_DIR,
    "lexicons": LEGACY_LEXICONS_DIR,
    "policy_bundles": LEGACY_POLICY_BUNDLES_DIR,
}
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemma2:2b")

_PYPROJECT = HERE.parent / "pyproject.toml"
with open(_PYPROJECT, "rb") as _f:
    APP_VERSION: str = tomllib.load(_f)["project"]["version"]
