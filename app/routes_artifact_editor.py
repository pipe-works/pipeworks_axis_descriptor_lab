"""
Artifact Editor routes.

These endpoints power the first-cut Artifact Editor page.  The current scope
is intentionally narrow:

- local prompt artifact browsing and loading
- local draft prompt creation with create-only safety rules
- local deterministic JSON artifact browsing and draft creation
- server-backed prompt manifests derived from the mud server's canonical lab
  endpoints

The route layer stays thin and delegates all file policy and manifest shaping
to ``app.artifact_editor``.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException

from app.artifact_editor import (
    create_local_axis_payload_draft,
    create_local_lexicon_json_draft,
    create_local_policy_bundle_draft,
    create_local_prompt_draft,
    get_server_prompt_manifest,
    list_local_axis_payload_artifacts,
    list_local_lexicon_json_artifacts,
    list_local_policy_bundle_artifacts,
    list_local_prompt_artifacts,
    load_local_axis_payload_artifact,
    load_local_lexicon_json_artifact,
    load_local_policy_bundle_artifact,
    load_local_prompt_artifact,
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
    ServerPromptManifestResponse,
)

router = APIRouter(tags=["artifact-editor"])


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

    The first cut is read-only: the lab consumes the mud server's world prompt
    files and contract metadata but does not write any server files.
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
