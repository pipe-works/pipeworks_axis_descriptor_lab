"""
Character-description save and export routes.

This router owns the default-system-prompt, save, and save-export endpoints.
It keeps HTTP concerns in the route layer while delegating save-folder
creation and file writing to the service layer.
"""

from __future__ import annotations

import io
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.config import DATA_DIR as _DATA_DIR
from app.file_loaders import load_default_prompt
from app.save_package import create_zip_archive
from app.schema import SaveRequest, SaveResponse
from app.services.save_service import save_run as persist_save_run

router = APIRouter(tags=["save"])

_FOLDER_NAME_PATTERN = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{8}$")


@router.get(
    "/api/system-prompt",
    response_class=PlainTextResponse,
    summary="Return the default system prompt text",
)
def get_system_prompt() -> str:
    """
    Return the default Character Description prompt as plain text.

    This allows the frontend to resolve the effective system prompt when no
    custom override is set, ensuring saved files always contain the complete
    prompt text rather than a placeholder.
    """
    return load_default_prompt()


@router.post(
    "/api/save",
    response_model=SaveResponse,
    summary="Save current session state to a timestamped subfolder under data/",
)
def save_run(req: SaveRequest) -> SaveResponse:
    """Persist a character-description session under the configured save root."""
    return persist_save_run(req, _DATA_DIR)


@router.get(
    "/api/save/{folder_name}/export",
    summary="Download a save package as a zip file",
)
def export_save(folder_name: str) -> StreamingResponse:
    """
    Stream a save folder as a compressed zip archive for download.

    The endpoint validates the folder name against a strict pattern to
    prevent path-traversal attacks, then bundles all known save-package
    files into a zip using ``save_package.create_zip_archive()``.
    """
    if not _FOLDER_NAME_PATTERN.match(folder_name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid folder name format: '{folder_name}'.",
        )

    save_dir = _DATA_DIR / folder_name
    if not save_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Save folder not found: '{folder_name}'.",
        )

    zip_bytes = create_zip_archive(save_dir)

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{folder_name}.zip"',
        },
    )
