"""
tools/dev_server.py
-----------------------------------------------------------------------------
Local dev launcher that loads .env before starting uvicorn.

Usage:
  python tools/dev_server.py
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")


def main() -> None:
    _load_env()

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8242"))
    reload = os.getenv("APP_RELOAD", "1").strip().lower() not in {"0", "false", "no"}

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
