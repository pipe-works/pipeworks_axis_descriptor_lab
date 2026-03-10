"""
tools/dev_server.py
-----------------------------------------------------------------------------
Local dev launcher that loads .env before starting uvicorn.

Usage:
  python tools/dev_server.py
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from uvicorn.config import LOGGING_CONFIG

SERVICE_LOG_LABEL = "axis-lab"


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")


def _build_uvicorn_log_config(service_label: str = SERVICE_LOG_LABEL) -> dict[str, Any]:
    """Build Uvicorn logging config with a service prefix."""
    log_config = deepcopy(LOGGING_CONFIG)
    formatters = log_config.get("formatters")
    if isinstance(formatters, dict):
        default_formatter = formatters.get("default")
        if isinstance(default_formatter, dict):
            default_formatter["fmt"] = f"{service_label} %(levelprefix)s %(message)s"

        access_formatter = formatters.get("access")
        if isinstance(access_formatter, dict):
            access_formatter["fmt"] = (
                f'{service_label} %(levelprefix)s %(client_addr)s - "%(request_line)s" '
                "%(status_code)s"
            )
    return log_config


def main() -> None:
    _load_env()

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8242"))
    reload = os.getenv("APP_RELOAD", "1").strip().lower() not in {"0", "false", "no"}

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_config=_build_uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
