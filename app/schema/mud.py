"""
Schemas for mud-server proxy requests and responses.

These models cover the small request/response surface used when the chat page
is connected to a Pipe-Works mud server instead of running in standalone
local-Ollama mode.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MudLoginRequest(BaseModel):
    """Request body for ``POST /api/mud/login``."""

    username: str = Field(..., description="Mud server account username.")
    password: str = Field(..., description="Mud server account password.")


class MudLoginResponse(BaseModel):
    """Response body for ``POST /api/mud/login``."""

    authenticated: bool = Field(
        ...,
        description="True when the mud server accepted the credentials.",
    )
    role: str | None = Field(
        default=None,
        description="User role on the mud server (e.g. 'admin', 'superuser').",
    )
    message: str | None = Field(
        default=None,
        description="Human-readable status message from the mud server.",
    )


class MudSessionResponse(BaseModel):
    """Response body for ``GET /api/mud/session``."""

    authenticated: bool = Field(
        ...,
        description="True when the lab holds a valid mud server session.",
    )
    role: str | None = Field(
        default=None,
        description="Cached role from the last successful login.",
    )
    translation_mode: str = Field(
        ...,
        description="Current translation mode: 'server-prod', 'server-local', or 'standalone'.",
    )


class MudSelectWorldRequest(BaseModel):
    """Request body for ``POST /api/mud/select-world``."""

    world_id: str = Field(..., description="World ID to select for translation.")
