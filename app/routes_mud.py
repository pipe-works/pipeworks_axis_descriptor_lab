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
    get_mud_mode_config,
    get_mud_client,
    set_mud_mode,
)
from app.schema import (
    MudLoginRequest,
    MudLoginResponse,
    MudModeRequest,
    MudModeResponse,
    MudSelectWorldRequest,
    MudSessionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mud"])
_LAB_ALLOWED_ROLES = frozenset({"admin", "superuser"})


def _is_lab_authorized_role(role: str | None) -> bool:
    """Return True when the mud-server role is allowed to use the lab UI."""
    return role in _LAB_ALLOWED_ROLES


@router.get("/mode", response_model=MudModeResponse, summary="Get runtime chat mode")
def mud_mode() -> MudModeResponse:
    """Return the active runtime chat mode and available mode options."""
    return MudModeResponse.model_validate(get_mud_mode_config())


@router.post("/mode", response_model=MudModeResponse, summary="Set runtime chat mode")
def mud_set_mode(req: MudModeRequest) -> MudModeResponse:
    """Switch the active chat translation mode without restarting the app."""
    try:
        return MudModeResponse.model_validate(set_mud_mode(req.mode_key, server_url=req.server_url))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/login", response_model=MudLoginResponse, summary="Proxy login to mud server")
def mud_login(req: MudLoginRequest) -> MudLoginResponse:
    """Proxy login to the mud server and store the session in memory."""
    client = get_mud_client()
    if client is None:
        return MudLoginResponse(
            authenticated=False,
            message="Offline mode active — switch to a server mode to connect.",
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

    role = data.get("role")
    if data.get("success") and not _is_lab_authorized_role(role):
        client.logout()
        return MudLoginResponse(
            authenticated=False,
            role=role,
            message="This mud server account is not authorised for the Axis Lab. Admin or superuser access is required.",
        )

    return MudLoginResponse(
        authenticated=data.get("success", False),
        role=role,
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
    mode = get_mud_mode_config()
    client = get_mud_client()
    if client is None:
        return MudSessionResponse(
            authenticated=False,
            selected_world_id=None,
            mode_key=mode["mode_key"],
            translation_mode=mode["translation_mode"],
            active_server_url=mode["active_server_url"],
        )
    status = client.session_status()
    return MudSessionResponse(
        authenticated=status["authenticated"],
        role=status.get("role"),
        selected_world_id=status.get("selected_world_id"),
        mode_key=mode["mode_key"],
        translation_mode=mode["translation_mode"],
        active_server_url=mode["active_server_url"],
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
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            raise HTTPException(
                status_code=403,
                detail="This mud server account is not authorised for the Axis Lab.",
            ) from exc
        raise
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
