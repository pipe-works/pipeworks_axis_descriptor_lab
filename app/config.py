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
_HOST_RUNTIME_ROOT = Path("/srv/work/pipeworks/runtime/axis-descriptor-lab")
_HOST_LOG_ROOT = Path("/srv/work/pipeworks/logs/axis-descriptor-lab")


def _path_from_env(env_var: str, host_default: Path, local_default: Path) -> Path:
    """Resolve a filesystem path from env, host baseline, or local fallback."""

    value = os.getenv(env_var)
    if value:
        return Path(value).expanduser()
    if host_default.parent.exists() and os.access(host_default.parent, os.W_OK):
        return host_default
    return local_default

DATA_DIR = _path_from_env("AXIS_LAB_DATA_DIR", _HOST_RUNTIME_ROOT, HERE.parent / "data")
LOGS_DIR = _path_from_env("AXIS_LAB_LOGS_DIR", _HOST_LOG_ROOT, HERE.parent / "logs")
WORLD_ROOT = HERE / "worlds"
LAB_ONLY_ROOT = HERE / "lab_only"
DEFAULT_WORLD_ID: str = os.getenv("LAB_DEFAULT_WORLD_ID", "pipeworks_web")
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemma2:2b")

_PYPROJECT = HERE.parent / "pyproject.toml"
with open(_PYPROJECT, "rb") as _f:
    APP_VERSION: str = tomllib.load(_f)["project"]["version"]
