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
import re
from typing import Any
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pipeworks_ipc import compute_payload_hash

from app.config import WORLD_ROOT
from app.mud_server_client import (
    MudServerConnectionError,
    MudServerSessionExpiredError,
    get_mud_mode_config,
    get_mud_client,
    set_mud_mode,
)
from app.schema import (
    MudCompileImagePromptRequest,
    MudImagePolicyBundleResponse,
    MudLoginRequest,
    MudLoginResponse,
    MudModeRequest,
    MudModeResponse,
    MudPipelineBootstrapResponse,
    MudPipelinePolicySource,
    MudPipelinePolicySourceReference,
    MudPipelineResolveRequest,
    MudPipelineResolveResponse,
    MudPipelineRuntimeOptions,
    MudPipelineSelectedBlocks,
    MudSelectWorldRequest,
    MudSessionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mud"])
_LAB_ALLOWED_ROLES = frozenset({"admin", "superuser"})


def _is_lab_authorized_role(role: str | None) -> bool:
    """Return True when the mud-server role is allowed to use the lab UI."""
    return role in _LAB_ALLOWED_ROLES


def _pipeline_error(status_code: int, *, detail: str, code: str, stage: str) -> JSONResponse:
    """Return a stable pipeline error payload.

    Pipeline-build frontend code needs stable machine-readable error fields so
    stage status updates are deterministic and independent from free-text
    messages. This helper keeps that contract shape consistent across endpoints.
    """

    return JSONResponse(
        status_code=status_code,
        content={
            "detail": detail,
            "code": code,
            "stage": stage,
        },
    )


def _extract_http_error_detail(exc: httpx.HTTPStatusError) -> str:
    """Return the most useful upstream HTTP error detail string."""

    response = exc.response
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        raw_detail = payload.get("detail")
        if isinstance(raw_detail, str) and raw_detail.strip():
            return raw_detail.strip()
        if raw_detail is not None:
            return str(raw_detail)

    text = response.text.strip()
    if text:
        return text[:500]
    return f"Upstream mud server returned HTTP {response.status_code}."


def _as_string_list(value: object) -> list[str]:
    """Normalize a candidate runtime-options value to a string list."""

    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


def _nested_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """Return nested dict value by key or an empty dict."""

    value = parent.get(key)
    return value if isinstance(value, dict) else {}


def _parse_inline_token_list(raw: str) -> list[str]:
    """Parse a ``[a, b, c]`` token list to normalized unique strings."""

    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    tokens = [part.strip() for part in text.split(",")]

    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        value = token.strip().strip("'\"")
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def _extract_species_from_local_policy_registry(
    world_id: str, world_root: Path = WORLD_ROOT
) -> list[str]:
    """Fallback species resolver from mirrored local policy files.

    Resolution path:
    1. ``<world>/policies/manifest.yaml`` -> ``image.registries.species``
    2. species registry entries -> ``compatible_species`` values
    """

    world_policies_root = world_root / world_id / "policies"
    manifest_path = world_policies_root / "manifest.yaml"
    if not manifest_path.is_file():
        return []

    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return []

    registry_rel_path: str | None = None
    for line in manifest_text.splitlines():
        match = re.match(r"^\s*species:\s*(\S.*)$", line)
        if not match:
            continue
        candidate = match.group(1).strip().strip("'\"")
        if candidate.endswith(".yaml") or candidate.endswith(".yml"):
            registry_rel_path = candidate
            break
    if not registry_rel_path:
        return []

    registry_path = world_root / world_id / registry_rel_path
    if not registry_path.is_file():
        return []

    try:
        registry_text = registry_path.read_text(encoding="utf-8")
    except OSError:
        return []

    species: list[str] = []
    seen: set[str] = set()
    for line in registry_text.splitlines():
        match = re.match(r"^\s*compatible_species:\s*(\[.*\])\s*$", line)
        if not match:
            continue
        for value in _parse_inline_token_list(match.group(1)):
            if value and value not in seen:
                species.append(value)
                seen.add(value)
    return species


def _extract_runtime_options(
    world_id: str,
    world_config: dict[str, Any],
) -> MudPipelineRuntimeOptions:
    """Extract runtime option sets from world config with policy-registry fallback."""

    image_generation = _nested_dict(world_config, "image_generation")
    runtime_options = _nested_dict(image_generation, "runtime_options")
    if not runtime_options:
        runtime_options = _nested_dict(world_config, "runtime_options")

    species = _as_string_list(runtime_options.get("species"))
    if not species:
        species = _extract_species_from_local_policy_registry(world_id)
    gender = _as_string_list(runtime_options.get("gender"))
    world_context_tags = _as_string_list(runtime_options.get("world_context_tags"))
    occupation_tags = _as_string_list(
        runtime_options.get("occupation_tags") or runtime_options.get("occupation_signals")
    )

    return MudPipelineRuntimeOptions(
        species=species,
        gender=gender or ["male", "female"],
        world_context_tags=world_context_tags,
        occupation_tags=occupation_tags,
    )


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


@router.get(
    "/world-image-policy-bundle/{world_id}",
    response_model=MudImagePolicyBundleResponse,
    summary="Get world image policy bundle",
)
def mud_world_image_policy_bundle(world_id: str) -> MudImagePolicyBundleResponse:
    """Proxy to ``GET /api/lab/world-image-policy-bundle/{world_id}`` on the mud server."""
    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    try:
        payload = client.world_image_policy_bundle(world_id)
        return MudImagePolicyBundleResponse.model_validate(payload)
    except MudServerSessionExpiredError:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        )
    except MudServerConnectionError:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.")


@router.get(
    "/pipeline-build/bootstrap/{world_id}",
    response_model=MudPipelineBootstrapResponse,
    summary="Aggregate canonical pipeline bootstrap metadata for one world",
)
def mud_pipeline_build_bootstrap(world_id: str) -> MudPipelineBootstrapResponse | JSONResponse:
    """Return stage-1/2 metadata in one payload for Pipeline Build.

    This endpoint intentionally aggregates multiple mud-server calls so the
    frontend can avoid orchestration race conditions across session/world/policy
    lookups.
    """

    client = get_mud_client()
    if client is None:
        return _pipeline_error(
            503,
            detail="Standalone mode — no mud server configured.",
            code="PIPELINE_MODE_UNAVAILABLE",
            stage="session_world",
        )

    try:
        worlds = client.list_worlds()
        world_row = next(
            (row for row in worlds if str(row.get("world_id") or "") == world_id),
            None,
        )
        if world_row is None:
            return _pipeline_error(
                404,
                detail=f"World {world_id!r} is not available for the active session.",
                code="PIPELINE_WORLD_NOT_FOUND",
                stage="session_world",
            )

        world_config = client.world_config(world_id)
        bundle_payload = client.world_image_policy_bundle(world_id)
        policy_bundle = MudImagePolicyBundleResponse.model_validate(bundle_payload)
        runtime_options = _extract_runtime_options(world_id, world_config)

        world_summary = {
            "world_row": world_row,
            "world_config": world_config,
        }
        return MudPipelineBootstrapResponse(
            world_id=world_id,
            world_summary=world_summary,
            policy_bundle=policy_bundle,
            policy_source=MudPipelinePolicySource(
                source_kind="mud_server_canonical",
                source_label="Mud server canonical",
                source_path=None,
                reference=MudPipelinePolicySourceReference(
                    world_id=world_id,
                    policy_bundle_id=policy_bundle.policy_bundle_id,
                    policy_bundle_version=policy_bundle.policy_bundle_version,
                    policy_hash=policy_bundle.policy_hash,
                ),
            ),
            runtime_options=runtime_options,
            required_fields=list(policy_bundle.required_runtime_inputs),
        )
    except MudServerSessionExpiredError:
        return _pipeline_error(
            401,
            detail="Mud server session expired. Please log in again.",
            code="PIPELINE_AUTH_REQUIRED",
            stage="session_world",
        )
    except MudServerConnectionError:
        return _pipeline_error(
            502,
            detail="Cannot connect to mud server.",
            code="PIPELINE_UPSTREAM_UNAVAILABLE",
            stage="session_world",
        )
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        detail = _extract_http_error_detail(exc)
        stage = "policy_bundle" if "policy" in detail.lower() else "session_world"
        code = (
            "PIPELINE_POLICY_UNAVAILABLE"
            if stage == "policy_bundle"
            else "PIPELINE_WORLD_LOAD_FAILED"
        )
        return _pipeline_error(status_code, detail=detail, code=code, stage=stage)
    except ValueError as exc:
        return _pipeline_error(
            502,
            detail=f"Invalid upstream payload for pipeline bootstrap: {exc}",
            code="PIPELINE_UPSTREAM_INVALID",
            stage="policy_bundle",
        )


@router.post(
    "/pipeline-build/resolve-image-selection",
    response_model=MudPipelineResolveResponse,
    summary="Resolve selected policy components for stages 5-7 without prompt output",
)
def mud_pipeline_build_resolve_image_selection(
    req: MudPipelineResolveRequest,
) -> MudPipelineResolveResponse | JSONResponse:
    """Return canonical selection metadata and deterministic pre-compile hashes.

    The route delegates selection to canonical mud-server compile logic and then
    strips final prompt text, returning only stage-5/6/7 metadata for the UI.
    """

    client = get_mud_client()
    if client is None:
        return _pipeline_error(
            503,
            detail="Standalone mode — no mud server configured.",
            code="PIPELINE_MODE_UNAVAILABLE",
            stage="session_world",
        )

    stage = "policy_bundle"
    try:
        bundle_payload = client.world_image_policy_bundle(req.world_id)
        policy_bundle = MudImagePolicyBundleResponse.model_validate(bundle_payload)

        stage = "compile_output"
        axes_payload = {
            axis_name: axis_value.model_dump() for axis_name, axis_value in req.axes.items()
        }
        compile_payload = client.compile_image_prompt(
            world_id=req.world_id,
            species=req.species,
            gender=req.gender,
            axes=axes_payload,
            world_context=req.world_context,
            occupation_signals=req.occupation_signals,
        )

        selected_species_block = compile_payload.get("selected_species_block_id")
        selected_clothing_slots = compile_payload.get("selected_clothing_slot_ids")
        if not isinstance(selected_clothing_slots, dict):
            selected_clothing_slots = {}

        composition_order = compile_payload.get("composition_order")
        if not isinstance(composition_order, list):
            composition_order = list(policy_bundle.composition_order)

        policy_hash = str(compile_payload.get("policy_hash") or policy_bundle.policy_hash)
        axis_hash_raw = compile_payload.get("axis_hash")
        if not isinstance(axis_hash_raw, str) or not axis_hash_raw.strip():
            return _pipeline_error(
                502,
                detail="Upstream compile response missing axis_hash.",
                code="PIPELINE_UPSTREAM_INVALID",
                stage="compile_output",
            )
        axis_hash = axis_hash_raw.strip()

        compiler_input_hash = compute_payload_hash(
            {
                "world_id": req.world_id,
                "identity": {"species": req.species, "gender": req.gender},
                "axes": axes_payload,
                "world_context": list(req.world_context),
                "occupation_signals": list(req.occupation_signals),
                "composition_order": composition_order,
                "policy_hash": policy_hash,
                "selected_blocks": {
                    "species_canon_block": selected_species_block,
                    "clothing_block": selected_clothing_slots,
                },
                "descriptor_layer": compile_payload.get("selected_descriptor_layer_id"),
                "tone_profile": compile_payload.get("selected_tone_profile_id"),
            }
        )

        return MudPipelineResolveResponse(
            selected_blocks=MudPipelineSelectedBlocks(
                species_canon_block=str(selected_species_block) if selected_species_block else None,
                clothing_block={key: value for key, value in selected_clothing_slots.items()},
            ),
            descriptor_layer=(
                str(compile_payload.get("selected_descriptor_layer_id"))
                if compile_payload.get("selected_descriptor_layer_id") is not None
                else None
            ),
            tone_profile=(
                str(compile_payload.get("selected_tone_profile_id"))
                if compile_payload.get("selected_tone_profile_id") is not None
                else None
            ),
            composition_order=[str(part) for part in composition_order],
            policy_hash=policy_hash,
            axis_hash=axis_hash,
            compiler_input_hash=compiler_input_hash,
        )
    except MudServerSessionExpiredError:
        return _pipeline_error(
            401,
            detail="Mud server session expired. Please log in again.",
            code="PIPELINE_AUTH_REQUIRED",
            stage="session_world",
        )
    except MudServerConnectionError:
        return _pipeline_error(
            502,
            detail="Cannot connect to mud server.",
            code="PIPELINE_UPSTREAM_UNAVAILABLE",
            stage=stage,
        )
    except httpx.HTTPStatusError as exc:
        return _pipeline_error(
            exc.response.status_code,
            detail=_extract_http_error_detail(exc),
            code="PIPELINE_UPSTREAM_HTTP_ERROR",
            stage=stage,
        )
    except ValueError as exc:
        return _pipeline_error(
            502,
            detail=f"Invalid upstream payload for pipeline resolve: {exc}",
            code="PIPELINE_UPSTREAM_INVALID",
            stage=stage,
        )


@router.post("/compile-image-prompt", summary="Compile canonical image prompt via mud server")
def mud_compile_image_prompt(req: MudCompileImagePromptRequest) -> dict:
    """Proxy ``POST /api/lab/compile-image-prompt`` using the active mud session."""
    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    try:
        axes_payload = {
            axis_name: axis_value.model_dump() for axis_name, axis_value in req.axes.items()
        }
        return client.compile_image_prompt(
            world_id=req.world_id,
            species=req.species,
            gender=req.gender,
            axes=axes_payload,
            world_context=req.world_context,
            occupation_signals=req.occupation_signals,
            model_id=req.model_id,
            aspect_ratio=req.aspect_ratio,
            seed=req.seed,
        )
    except MudServerSessionExpiredError:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail=_extract_http_error_detail(exc)
        ) from exc
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
