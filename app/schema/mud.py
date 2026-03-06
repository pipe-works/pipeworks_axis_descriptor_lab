"""
Schemas for mud-server proxy requests and responses.

These models cover the small request/response surface used when the chat page
is connected to a Pipe-Works mud server instead of running in standalone
local-Ollama mode.  They also expose the runtime mode selector that lets the
frontend switch between offline and server-backed chat translation without
editing environment files.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schema.axis import AxisValue


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
    selected_world_id: str | None = Field(
        default=None,
        description="Currently selected world ID for the active mud server session, if any.",
    )
    mode_key: str = Field(
        ...,
        description="Current runtime chat mode selector key: 'standalone', 'development', or 'configured'.",
    )
    translation_mode: str = Field(
        ...,
        description="Current translation mode: 'server-prod', 'server-local', or 'standalone'.",
    )
    active_server_url: str | None = Field(
        default=None,
        description="Active mud server base URL for the selected runtime mode, or null in offline mode.",
    )


class MudModeOption(BaseModel):
    """One runtime-selectable chat translation mode."""

    key: str = Field(..., description="Stable selector key for the mode option.")
    label: str = Field(..., description="Human-readable mode label shown in the chat UI.")
    translation_mode: str = Field(
        ...,
        description="Public translation mode string associated with this option.",
    )
    server_url: str | None = Field(
        default=None,
        description="Mud server base URL for this option, or null for offline mode.",
    )


class MudModeRequest(BaseModel):
    """Request body for ``POST /api/mud/mode``."""

    mode_key: str = Field(
        ...,
        description="Runtime mode selector key to activate for the chat page.",
    )
    server_url: str | None = Field(
        default=None,
        description=(
            "Optional mud server URL override for development mode. Ignored for "
            "offline and configured-server modes."
        ),
    )


class MudModeResponse(BaseModel):
    """Response body for ``GET`` and ``POST`` ``/api/mud/mode``."""

    mode_key: str = Field(..., description="Currently active runtime mode selector key.")
    translation_mode: str = Field(
        ...,
        description="Current translation mode: 'server-prod', 'server-local', or 'standalone'.",
    )
    active_server_url: str | None = Field(
        default=None,
        description="Active mud server base URL for the current mode, or null in offline mode.",
    )
    available_modes: list[MudModeOption] = Field(
        ...,
        description="All mode options the current lab process can switch between at runtime.",
    )


class MudSelectWorldRequest(BaseModel):
    """Request body for ``POST /api/mud/select-world``."""

    world_id: str = Field(..., description="World ID to select for translation.")


class MudCompileImagePromptRequest(BaseModel):
    """Request body for ``POST /api/mud/compile-image-prompt``.

    This mirrors the mud server's canonical image compile request while
    intentionally keeping the surface minimal for the lab's phase-1
    canonical mode.
    """

    world_id: str = Field(..., description="Target world ID on the mud server.")
    species: str = Field(
        default="goblin",
        description="Species identifier used by the mud server's species block selector.",
    )
    gender: str = Field(
        default="male",
        description="Fixed identity gender value expected by canonical image policy.",
    )
    axes: dict[str, AxisValue] = Field(
        ...,
        description="Axis name to label/score mapping forwarded to canonical compile.",
    )
    world_context: list[str] = Field(
        default_factory=list,
        description="Optional world context tags used by clothing selection rules.",
    )
    occupation_signals: list[str] = Field(
        default_factory=list,
        description="Optional occupation/activity tags used by clothing selection rules.",
    )
    model_id: str | None = Field(
        default=None,
        description="Optional generation model hint forwarded to canonical compile.",
    )
    aspect_ratio: str | None = Field(
        default=None,
        description="Optional aspect-ratio hint forwarded to canonical compile.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional generation seed hint forwarded to canonical compile.",
    )


class MudImagePolicyBundleResponse(BaseModel):
    """Response body for ``GET /api/mud/world-image-policy-bundle/{world_id}``."""

    world_id: str = Field(..., description="World identifier.")
    policy_schema: str | None = Field(default=None, description="Manifest policy schema id.")
    policy_bundle_id: str | None = Field(default=None, description="Active policy bundle id.")
    policy_bundle_version: int | str | None = Field(
        default=None, description="Active policy bundle version."
    )
    policy_hash: str = Field(..., description="Deterministic hash of compiler policy inputs.")
    composition_order: list[str] = Field(default_factory=list)
    required_runtime_inputs: list[str] = Field(default_factory=list)
    missing_components: list[str] = Field(default_factory=list)
