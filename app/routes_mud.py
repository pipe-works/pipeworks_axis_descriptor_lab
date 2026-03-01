"""
Mud-server proxy routes for the Axis Descriptor Lab.

This router owns the ``/api/mud/*`` endpoints that proxy authentication,
session, and world-selection requests to the optional Pipe-Works mud server.
The logic here is intentionally thin: validate the local mode, delegate to
the shared mud client, and translate connection/session failures into stable
HTTP responses for the frontend.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException

from app.mud_server_client import (
    MudServerConnectionError,
    MudServerSessionExpiredError,
    compute_translation_mode,
    get_mud_client,
)
from app.schema import (
    MudLoginRequest,
    MudLoginResponse,
    MudSelectWorldRequest,
    MudSessionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mud"])


@router.post("/login", response_model=MudLoginResponse, summary="Proxy login to mud server")
def mud_login(req: MudLoginRequest) -> MudLoginResponse:
    """Proxy login to the mud server and store the session in memory."""
    client = get_mud_client()
    if client is None:
        return MudLoginResponse(
            authenticated=False,
            message="Standalone mode — no mud server configured.",
        )
    try:
        data = client.login(req.username, req.password)
    except MudServerConnectionError:
        return MudLoginResponse(authenticated=False, message="Cannot connect to mud server.")
    except httpx.HTTPStatusError as exc:
        return MudLoginResponse(
            authenticated=False,
            message=f"Login failed: {exc.response.status_code}",
        )

    return MudLoginResponse(
        authenticated=data.get("success", False),
        role=data.get("role"),
        message=data.get("message"),
    )


@router.post("/logout", summary="Clear mud server session")
def mud_logout() -> dict:
    """Clear the mud server session from memory."""
    client = get_mud_client()
    if client is not None:
        client.logout()
    return {"success": True}


@router.get("/session", response_model=MudSessionResponse, summary="Auth status")
def mud_session() -> MudSessionResponse:
    """Return current mud server auth status and translation mode."""
    client = get_mud_client()
    if client is None:
        return MudSessionResponse(
            authenticated=False,
            translation_mode=compute_translation_mode(),
        )
    status = client.session_status()
    return MudSessionResponse(
        authenticated=status["authenticated"],
        role=status.get("role"),
        translation_mode=compute_translation_mode(),
    )


@router.get("/worlds", summary="List mud server worlds")
def mud_worlds() -> dict:
    """Proxy to ``GET /api/lab/worlds`` on the mud server."""
    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    try:
        worlds = client.list_worlds()
        for world in worlds:
            logger.debug("mud_worlds: %r", world)
        return {"worlds": worlds}
    except MudServerSessionExpiredError:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        )
    except MudServerConnectionError:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.")


@router.get("/world-config/{world_id}", summary="Get world config")
def mud_world_config(world_id: str) -> dict:
    """Proxy to ``GET /api/lab/world-config/{world_id}`` on the mud server."""
    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    try:
        return client.world_config(world_id)
    except MudServerSessionExpiredError:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        )
    except MudServerConnectionError:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.")


@router.get("/world-prompts/{world_id}", summary="Get world prompt templates")
def mud_world_prompts(world_id: str) -> dict:
    """Proxy to ``GET /api/lab/world-prompts/{world_id}`` on the mud server."""
    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    try:
        return client.world_prompts(world_id)
    except MudServerSessionExpiredError:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        )
    except MudServerConnectionError:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.")


@router.post("/select-world", summary="Select world for translation")
def mud_select_world(req: MudSelectWorldRequest) -> dict:
    """Store the selected world_id in the MudServerClient's memory."""
    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    client.select_world(req.world_id)
    return {"success": True, "world_id": req.world_id}
