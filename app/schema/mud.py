"""
Schemas for mud-server proxy requests and responses.

These models cover the small request/response surface used when the chat page
is connected to a Pipe-Works mud server instead of running in standalone
local-Ollama mode.  They also expose the runtime mode selector that lets the
frontend switch between offline and server-backed chat translation without
editing environment files.
"""

from __future__ import annotations

from typing import Any, Literal

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


class MudPipelineGenerateConditionAxisRequest(BaseModel):
    """Request body for ``POST /api/mud/pipeline-build/generate-condition-axis``.

    This contract is intentionally minimal for production canonical mode:
    the mud server owns condition-axis generation and returns an AxisPayload.
    ``seed`` may be omitted (or null) to request server-side random seeding.
    """

    world_id: str = Field(
        ...,
        description="Target world ID used for canonical condition-axis generation.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional deterministic seed; null delegates to server random seed generation.",
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


class MudPipelineRuntimeOptions(BaseModel):
    """Runtime option sets used by Pipeline Build stage controls.

    The bootstrap endpoint uses this model to expose optional selector values
    without forcing the frontend to infer them from world configuration files.
    """

    species: list[str] = Field(
        default_factory=list,
        description="Allowed species identifiers for image compile/selection.",
    )
    gender: list[str] = Field(
        default_factory=lambda: ["male", "female"],
        description="Allowed gender values for phase-1 image policy.",
    )
    world_context_tags: list[str] = Field(
        default_factory=list,
        description="Known world-context tags accepted by clothing selection.",
    )
    occupation_tags: list[str] = Field(
        default_factory=list,
        description="Known occupation/activity tags accepted by clothing selection.",
    )


class MudPipelinePolicySourceReference(BaseModel):
    """Canonical reference metadata for one resolved policy source.

    This model deliberately avoids filesystem assumptions for remote policy
    sources. It identifies the policy bundle by deterministic IDs and hashes
    so UI surfaces can remain truthful across local vs remote environments.
    """

    world_id: str = Field(..., description="World identifier used for policy lookup.")
    policy_bundle_id: str | None = Field(default=None, description="Resolved policy bundle id.")
    policy_bundle_version: int | str | None = Field(
        default=None, description="Resolved policy bundle version."
    )
    policy_hash: str = Field(..., description="Deterministic policy hash for the active bundle.")
    served_via: str = Field(
        default="/api/mud/pipeline-build/bootstrap/{world_id}",
        description="Endpoint that served this policy source reference.",
    )


class MudPipelinePolicySource(BaseModel):
    """Truthful source metadata for Policy Bundle display surfaces."""

    source_kind: Literal[
        "mud_server_canonical",
        "local_world",
        "lab_only",
        "legacy",
        "offline",
        "unknown",
    ] = Field(..., description="Resolved policy source kind token.")
    source_label: str = Field(..., description="Human-readable label for the source kind.")
    source_path: str | None = Field(
        default=None,
        description="Filesystem path when source is local; null for remote canonical sources.",
    )
    reference: MudPipelinePolicySourceReference | None = Field(
        default=None,
        description="Canonical policy reference fields for reproducibility and diagnostics.",
    )


class MudPipelineBootstrapResponse(BaseModel):
    """Response body for ``GET /api/mud/pipeline-build/bootstrap/{world_id}``.

    This aggregated payload keeps the frontend stateless by returning all stage-1
    and stage-2 inputs in one call.
    """

    world_id: str = Field(..., description="World identifier used for bootstrap.")
    world_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved world row + world config summary metadata.",
    )
    policy_bundle: MudImagePolicyBundleResponse = Field(
        ...,
        description="Canonical image policy bundle metadata for the world.",
    )
    policy_source: MudPipelinePolicySource | None = Field(
        default=None,
        description="Truthful source metadata for policy bundle origin display.",
    )
    runtime_options: MudPipelineRuntimeOptions = Field(
        default_factory=MudPipelineRuntimeOptions,
        description="Runtime option sets for stage selectors.",
    )
    required_fields: list[str] = Field(
        default_factory=list,
        description="Runtime compile contract fields required by policy composition.",
    )


class MudPipelineResolveRequest(BaseModel):
    """Request body for ``POST /api/mud/pipeline-build/resolve-image-selection``.

    The request intentionally excludes generation-only controls so the response
    can represent pre-prompt selection/resolve state for stages 5-7.
    """

    world_id: str = Field(..., description="Target world ID on the mud server.")
    species: str = Field(
        default="goblin",
        description="Species identifier used for species block selection.",
    )
    gender: str = Field(
        default="male",
        description="Identity gender value expected by policy selection rules.",
    )
    axes: dict[str, AxisValue] = Field(
        ...,
        description="Axis label/score payload used for deterministic selection.",
    )
    world_context: list[str] = Field(
        default_factory=list,
        description="Optional world-context tags used by clothing selection.",
    )
    occupation_signals: list[str] = Field(
        default_factory=list,
        description="Optional occupation/activity tags used by clothing selection.",
    )


class MudPipelineSelectedBlocks(BaseModel):
    """Selected block IDs returned by the resolve preview endpoint."""

    species_canon_block: str | None = Field(
        default=None,
        description="Selected species canon block entry id.",
    )
    clothing_block: dict[str, str | None] = Field(
        default_factory=dict,
        description="Selected clothing block entry ids keyed by clothing slot.",
    )


class MudPipelineResolveResponse(BaseModel):
    """Response body for ``POST /api/mud/pipeline-build/resolve-image-selection``.

    This response intentionally omits the compiled prompt text so stage-5/6/7
    UI state can be rendered without coupling to stage-8 side effects.
    """

    selected_blocks: MudPipelineSelectedBlocks = Field(
        ...,
        description="Selected species/clothing blocks for the current runtime inputs.",
    )
    descriptor_layer: str | None = Field(
        default=None,
        description="Resolved descriptor layer identifier.",
    )
    tone_profile: str | None = Field(
        default=None,
        description="Resolved tone profile identifier.",
    )
    composition_order: list[str] = Field(
        default_factory=list,
        description="Composition order used by canonical prompt assembly.",
    )
    policy_hash: str = Field(..., description="Deterministic hash of policy compiler inputs.")
    axis_hash: str = Field(..., description="Deterministic hash of axis runtime payload.")
    compiler_input_hash: str = Field(
        ...,
        description="Deterministic hash of normalized compiler inputs, excluding compiled prompt text.",
    )
