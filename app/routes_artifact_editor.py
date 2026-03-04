"""
Artifact Editor routes.

These endpoints power the first-cut Artifact Editor page.  The current scope
is intentionally narrow:

- local prompt artifact browsing and loading
- local draft prompt creation with create-only safety rules
- local deterministic JSON artifact browsing and draft creation
- server-backed prompt manifests derived from the mud server's canonical lab
  endpoints
- server-backed prompt draft listing, loading, and create-only draft creation
- server-backed prompt draft promotion through explicit mud-server APIs
- server-backed policy bundle inspection, create-only draft creation, and
  explicit canonical promotion

The route layer stays thin and delegates all file policy and manifest shaping
to ``app.artifact_editor``.
"""

from __future__ import annotations

from typing import Literal, NoReturn

import httpx
from fastapi import APIRouter, HTTPException

from app.artifact_editor import (
    create_server_policy_bundle_draft,
    create_server_prompt_draft,
    create_local_axis_payload_draft,
    create_local_lexicon_json_draft,
    create_local_policy_bundle_draft,
    create_local_prompt_draft,
    get_server_policy_bundle_artifact,
    get_server_prompt_manifest,
    list_server_prompt_artifacts,
    list_server_policy_bundle_artifacts,
    list_local_axis_payload_artifacts,
    list_local_lexicon_json_artifacts,
    list_local_policy_bundle_artifacts,
    list_local_prompt_artifacts,
    load_server_policy_bundle_draft_artifact,
    load_server_prompt_draft_artifact,
    load_local_axis_payload_artifact,
    load_local_lexicon_json_artifact,
    load_local_policy_bundle_artifact,
    load_local_prompt_artifact,
    promote_server_policy_bundle_draft,
    promote_server_prompt_draft,
)
from app.mud_server_client import (
    MudServerConnectionError,
    MudServerSessionExpiredError,
    get_mud_client,
)
from app.schema import (
    AxisPayloadArtifactDocument,
    LocalAxisPayloadArtifactListResponse,
    LocalAxisPayloadDraftCreateRequest,
    LocalAxisPayloadDraftCreateResponse,
    LexiconJsonArtifactDocument,
    LocalLexiconJsonArtifactListResponse,
    LocalLexiconJsonDraftCreateRequest,
    LocalLexiconJsonDraftCreateResponse,
    LocalPolicyBundleArtifactListResponse,
    LocalPolicyBundleDraftCreateRequest,
    LocalPolicyBundleDraftCreateResponse,
    LocalPromptArtifactListResponse,
    LocalPromptDraftCreateRequest,
    LocalPromptDraftCreateResponse,
    PolicyBundleArtifactDocument,
    PromptArtifactDocument,
    ServerPromptArtifactListResponse,
    ServerPromptDraftCreateRequest,
    ServerPromptDraftCreateResponse,
    ServerPromptDraftPromoteRequest,
    ServerPromptDraftPromoteResponse,
    ServerPolicyBundleArtifactListResponse,
    ServerPolicyBundleDraftCreateRequest,
    ServerPolicyBundleDraftCreateResponse,
    ServerPolicyBundleDraftPromoteRequest,
    ServerPolicyBundleDraftPromoteResponse,
    ServerPromptManifestResponse,
)

router = APIRouter(tags=["artifact-editor"])


def _raise_for_mud_http_error(exc: httpx.HTTPStatusError) -> NoReturn:
    """Translate mud-server HTTP failures into stable lab-facing HTTP errors."""

    detail: str
    try:
        payload = exc.response.json()
        detail = str(payload.get("detail") or exc.response.text)
    except ValueError:
        detail = exc.response.text

    raise HTTPException(
        status_code=exc.response.status_code,
        detail=detail or "Mud server request failed.",
    ) from exc


@router.get(
    "/api/artifacts/local/chat-prompts",
    response_model=LocalPromptArtifactListResponse,
    summary="List local prompt artifacts for the Artifact Editor",
)
def list_local_chat_prompts(
    purpose: Literal["character_description", "chat_translation"],
) -> LocalPromptArtifactListResponse:
    """Return local prompt files, including drafts, for one prompt family."""

    return list_local_prompt_artifacts(purpose)


@router.get(
    "/api/artifacts/local/chat-prompts/{name}",
    response_model=PromptArtifactDocument,
    summary="Load one local prompt artifact",
)
def get_local_chat_prompt(
    name: str,
    purpose: Literal["character_description", "chat_translation"],
) -> PromptArtifactDocument:
    """Load one prompt file together with its editor reference contract."""

    return load_local_prompt_artifact(name, purpose)


@router.post(
    "/api/artifacts/local/chat-prompts/drafts",
    response_model=LocalPromptDraftCreateResponse,
    summary="Create a new local draft prompt artifact",
)
def create_local_chat_prompt_draft(
    req: LocalPromptDraftCreateRequest,
) -> LocalPromptDraftCreateResponse:
    """Create a new draft prompt file under the local prompt tree."""

    return create_local_prompt_draft(req)


@router.get(
    "/api/artifacts/local/axis-payloads",
    response_model=LocalAxisPayloadArtifactListResponse,
    summary="List local AxisPayload JSON artifacts for the Artifact Editor",
)
def list_local_axis_payloads() -> LocalAxisPayloadArtifactListResponse:
    """Return local AxisPayload JSON files, including drafts."""

    return list_local_axis_payload_artifacts()


@router.get(
    "/api/artifacts/local/axis-payloads/{name}",
    response_model=AxisPayloadArtifactDocument,
    summary="Load one local AxisPayload JSON artifact",
)
def get_local_axis_payload(name: str) -> AxisPayloadArtifactDocument:
    """Load one AxisPayload JSON file together with its editor reference contract."""

    return load_local_axis_payload_artifact(name)


@router.post(
    "/api/artifacts/local/axis-payloads/drafts",
    response_model=LocalAxisPayloadDraftCreateResponse,
    summary="Create a new local AxisPayload JSON draft",
)
def create_local_axis_payload_draft_route(
    req: LocalAxisPayloadDraftCreateRequest,
) -> LocalAxisPayloadDraftCreateResponse:
    """Create a new validated AxisPayload JSON draft file."""

    return create_local_axis_payload_draft(req)


@router.get(
    "/api/artifacts/local/lexicons",
    response_model=LocalLexiconJsonArtifactListResponse,
    summary="List local deterministic lexicon JSON artifacts for the Artifact Editor",
)
def list_local_lexicons() -> LocalLexiconJsonArtifactListResponse:
    """Return local lexicon JSON files, including drafts."""

    return list_local_lexicon_json_artifacts()


@router.get(
    "/api/artifacts/local/lexicons/{name}",
    response_model=LexiconJsonArtifactDocument,
    summary="Load one local deterministic lexicon JSON artifact",
)
def get_local_lexicon(name: str) -> LexiconJsonArtifactDocument:
    """Load one deterministic lexicon JSON file together with its reference contract."""

    return load_local_lexicon_json_artifact(name)


@router.post(
    "/api/artifacts/local/lexicons/drafts",
    response_model=LocalLexiconJsonDraftCreateResponse,
    summary="Create a new local deterministic lexicon JSON draft",
)
def create_local_lexicon_draft_route(
    req: LocalLexiconJsonDraftCreateRequest,
) -> LocalLexiconJsonDraftCreateResponse:
    """Create a new validated deterministic lexicon JSON draft file."""

    return create_local_lexicon_json_draft(req)


@router.get(
    "/api/artifacts/local/policy-bundles",
    response_model=LocalPolicyBundleArtifactListResponse,
    summary="List local normalized policy bundle JSON artifacts for the Artifact Editor",
)
def list_local_policy_bundles() -> LocalPolicyBundleArtifactListResponse:
    """Return local normalized policy bundle JSON files, including drafts."""

    return list_local_policy_bundle_artifacts()


@router.get(
    "/api/artifacts/local/policy-bundles/{name}",
    response_model=PolicyBundleArtifactDocument,
    summary="Load one local normalized policy bundle JSON artifact",
)
def get_local_policy_bundle(name: str) -> PolicyBundleArtifactDocument:
    """Load one normalized policy bundle JSON file together with its reference contract."""

    return load_local_policy_bundle_artifact(name)


@router.post(
    "/api/artifacts/local/policy-bundles/drafts",
    response_model=LocalPolicyBundleDraftCreateResponse,
    summary="Create a new local normalized policy bundle JSON draft",
)
def create_local_policy_bundle_draft_route(
    req: LocalPolicyBundleDraftCreateRequest,
) -> LocalPolicyBundleDraftCreateResponse:
    """Create a new validated normalized policy bundle JSON draft file."""

    return create_local_policy_bundle_draft(req)


@router.get(
    "/api/artifacts/server/chat-prompts/{world_id}",
    response_model=ServerPromptManifestResponse,
    summary="Get a server-backed canonical prompt manifest",
)
def get_server_chat_prompts(world_id: str) -> ServerPromptManifestResponse:
    """Return canonical world prompt data for the Artifact Editor.

    The mud server remains authoritative: this route exposes canonical prompt
    files only and never mutates server state.
    """

    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    if not client.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated — please log in.")

    try:
        return get_server_prompt_manifest(world_id, client)
    except MudServerSessionExpiredError as exc:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        ) from exc
    except MudServerConnectionError as exc:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.") from exc
    except httpx.HTTPStatusError as exc:
        _raise_for_mud_http_error(exc)


@router.get(
    "/api/artifacts/server/chat-prompts/{world_id}/drafts",
    response_model=ServerPromptArtifactListResponse,
    summary="List mud-server prompt drafts",
)
def list_server_chat_prompt_drafts(world_id: str) -> ServerPromptArtifactListResponse:
    """Return draft prompt files for one mud-server world."""

    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    if not client.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated — please log in.")

    try:
        return list_server_prompt_artifacts(world_id, client)
    except MudServerSessionExpiredError as exc:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        ) from exc
    except MudServerConnectionError as exc:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.") from exc
    except httpx.HTTPStatusError as exc:
        _raise_for_mud_http_error(exc)


@router.get(
    "/api/artifacts/server/chat-prompts/{world_id}/drafts/{name}",
    response_model=PromptArtifactDocument,
    summary="Load one mud-server prompt draft",
)
def get_server_chat_prompt_draft(world_id: str, name: str) -> PromptArtifactDocument:
    """Load one draft prompt file for the selected mud-server world."""

    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    if not client.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated — please log in.")

    try:
        return load_server_prompt_draft_artifact(
            world_id=world_id, draft_name=name, mud_client=client
        )
    except MudServerSessionExpiredError as exc:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        ) from exc
    except MudServerConnectionError as exc:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.") from exc
    except httpx.HTTPStatusError as exc:
        _raise_for_mud_http_error(exc)


@router.post(
    "/api/artifacts/server/chat-prompts/{world_id}/drafts",
    response_model=ServerPromptDraftCreateResponse,
    summary="Create a new mud-server prompt draft",
)
def create_server_chat_prompt_draft_route(
    world_id: str,
    req: ServerPromptDraftCreateRequest,
) -> ServerPromptDraftCreateResponse:
    """Create a new draft under the mud server's prompt draft directory."""

    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    if not client.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated — please log in.")

    try:
        return create_server_prompt_draft(world_id=world_id, req=req, mud_client=client)
    except MudServerSessionExpiredError as exc:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        ) from exc
    except MudServerConnectionError as exc:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.") from exc
    except httpx.HTTPStatusError as exc:
        _raise_for_mud_http_error(exc)


@router.post(
    "/api/artifacts/server/chat-prompts/{world_id}/drafts/{name}/promote",
    response_model=ServerPromptDraftPromoteResponse,
    summary="Promote one mud-server prompt draft to canonical status",
)
def promote_server_chat_prompt_draft_route(
    world_id: str,
    name: str,
    req: ServerPromptDraftPromoteRequest,
) -> ServerPromptDraftPromoteResponse:
    """Promote one mud-server prompt draft into a new canonical active prompt file."""

    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    if not client.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated — please log in.")

    try:
        return promote_server_prompt_draft(
            world_id=world_id,
            draft_name=name,
            req=req,
            mud_client=client,
        )
    except MudServerSessionExpiredError as exc:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        ) from exc
    except MudServerConnectionError as exc:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.") from exc
    except httpx.HTTPStatusError as exc:
        _raise_for_mud_http_error(exc)


@router.get(
    "/api/artifacts/server/policy-bundles/{world_id}",
    response_model=PolicyBundleArtifactDocument,
    summary="Get a server-backed canonical policy bundle artifact",
)
def get_server_policy_bundle(world_id: str) -> PolicyBundleArtifactDocument:
    """Return the mud server's normalized canonical policy bundle for one world."""

    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    if not client.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated — please log in.")

    try:
        return get_server_policy_bundle_artifact(world_id, client)
    except MudServerSessionExpiredError as exc:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        ) from exc
    except MudServerConnectionError as exc:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.") from exc
    except httpx.HTTPStatusError as exc:
        _raise_for_mud_http_error(exc)


@router.get(
    "/api/artifacts/server/policy-bundles/{world_id}/drafts",
    response_model=ServerPolicyBundleArtifactListResponse,
    summary="List mud-server policy bundle drafts",
)
def list_server_policy_bundle_drafts(world_id: str) -> ServerPolicyBundleArtifactListResponse:
    """Return draft policy bundle files for one mud-server world."""

    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    if not client.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated — please log in.")

    try:
        return list_server_policy_bundle_artifacts(world_id, client)
    except MudServerSessionExpiredError as exc:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        ) from exc
    except MudServerConnectionError as exc:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.") from exc
    except httpx.HTTPStatusError as exc:
        _raise_for_mud_http_error(exc)


@router.get(
    "/api/artifacts/server/policy-bundles/{world_id}/drafts/{name}",
    response_model=PolicyBundleArtifactDocument,
    summary="Load one mud-server policy bundle draft",
)
def get_server_policy_bundle_draft(world_id: str, name: str) -> PolicyBundleArtifactDocument:
    """Load one draft policy bundle file for the selected mud-server world."""

    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    if not client.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated — please log in.")

    try:
        return load_server_policy_bundle_draft_artifact(
            world_id=world_id,
            draft_name=name,
            mud_client=client,
        )
    except MudServerSessionExpiredError as exc:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        ) from exc
    except MudServerConnectionError as exc:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.") from exc
    except httpx.HTTPStatusError as exc:
        _raise_for_mud_http_error(exc)


@router.post(
    "/api/artifacts/server/policy-bundles/{world_id}/drafts",
    response_model=ServerPolicyBundleDraftCreateResponse,
    summary="Create a new mud-server policy bundle draft",
)
def create_server_policy_bundle_draft_route(
    world_id: str,
    req: ServerPolicyBundleDraftCreateRequest,
) -> ServerPolicyBundleDraftCreateResponse:
    """Create a new draft under the mud server's policy draft directory.

    The mud server remains authoritative: this route can only create new
    draft files and cannot overwrite active canonical policy files.
    """

    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    if not client.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated — please log in.")

    try:
        return create_server_policy_bundle_draft(world_id=world_id, req=req, mud_client=client)
    except MudServerSessionExpiredError as exc:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        ) from exc
    except MudServerConnectionError as exc:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.") from exc
    except httpx.HTTPStatusError as exc:
        _raise_for_mud_http_error(exc)


@router.post(
    "/api/artifacts/server/policy-bundles/{world_id}/drafts/{name}/promote",
    response_model=ServerPolicyBundleDraftPromoteResponse,
    summary="Promote one mud-server policy bundle draft to canonical status",
)
def promote_server_policy_bundle_draft_route(
    world_id: str,
    name: str,
    req: ServerPolicyBundleDraftPromoteRequest,
) -> ServerPolicyBundleDraftPromoteResponse:
    """Promote one mud-server policy bundle draft into canonical policy files."""

    client = get_mud_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Standalone mode — no mud server configured.")
    if not client.is_authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated — please log in.")

    try:
        return promote_server_policy_bundle_draft(
            world_id=world_id,
            draft_name=name,
            req=req,
            mud_client=client,
        )
    except MudServerSessionExpiredError as exc:
        raise HTTPException(
            status_code=401,
            detail="Mud server session expired. Please log in again.",
        ) from exc
    except MudServerConnectionError as exc:
        raise HTTPException(status_code=502, detail="Cannot connect to mud server.") from exc
    except httpx.HTTPStatusError as exc:
        _raise_for_mud_http_error(exc)
