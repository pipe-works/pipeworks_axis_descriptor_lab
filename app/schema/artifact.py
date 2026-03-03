"""
Schemas for the Artifact Editor page.

These models describe the Artifact Editor workflow across both prompt and
deterministic JSON artifacts:

- local prompt files stored in ``app/prompts/*`` plus draft files under
  ``app/prompts/*/drafts``
- server-backed prompt manifests derived from the mud server's canonical world
  prompt endpoints
- local AxisPayload JSON artifacts from ``app/examples``
- local normalized policy bundle JSON artifacts from
  ``app/artifacts/policy_bundles``
- local deterministic lexicon JSON artifacts from ``app/data``
- create-only local draft saves that avoid overwriting shipped files
- create-only mud-server prompt and policy-bundle draft saves that avoid
  overwriting canonical server files

The models are intentionally explicit so the frontend receives structured
metadata rather than reverse-engineering file paths and placeholder contracts
from free-form text.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ArtifactPlaceholder(BaseModel):
    """One supported template placeholder and its meaning.

    The editor uses these rows to build the reference panel and to validate
    placeholder usage in the raw prompt textarea.
    """

    placeholder: str = Field(
        ...,
        description="Placeholder token including braces, for example '{{channel}}'.",
    )
    description: str = Field(
        ...,
        description="Short explanation of where the placeholder value comes from.",
    )


class ArtifactPromptReference(BaseModel):
    """Prompt-contract metadata shown beside the raw text editor.

    This structure gives the Artifact Editor enough context to explain and
    validate a prompt template without hardcoding all assumptions inside the
    frontend.
    """

    source_mode: Literal["local", "server"] = Field(
        ...,
        description="Whether the contract came from local lab rules or a server-backed world.",
    )
    purpose: Literal["character_description", "chat_translation"] = Field(
        ...,
        description="Prompt family that owns the contract.",
    )
    world_id: str | None = Field(
        default=None,
        description="World identifier when the contract is server-backed, else null.",
    )
    active_axes: list[str] = Field(
        default_factory=list,
        description="Canonical active axis ordering used for axis-specific placeholders.",
    )
    placeholders: list[ArtifactPlaceholder] = Field(
        default_factory=list,
        description="Supported placeholders for this prompt contract.",
    )
    sample_values: dict[str, str] = Field(
        default_factory=dict,
        description="Example placeholder substitutions for preview rendering.",
    )
    profile_summary_example: str | None = Field(
        default=None,
        description="Example canonical profile_summary block, if the contract exposes one.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Human-readable constraints or caveats for this contract.",
    )


class PromptArtifactSummary(BaseModel):
    """Metadata for one selectable prompt artifact."""

    name: str = Field(..., description="Prompt stem without the .txt extension.")
    purpose: Literal["character_description", "chat_translation"] = Field(
        ...,
        description="Prompt family that owns the file.",
    )
    is_draft: bool = Field(
        ...,
        description="True when the file lives under a drafts/ directory rather than a shipped path.",
    )
    is_active: bool = Field(
        default=False,
        description="True when this prompt is the active canonical world prompt in server mode.",
    )
    origin_path: str = Field(
        ...,
        description="Path relative to the owning prompts directory or world policies directory.",
    )
    content: str | None = Field(
        default=None,
        description="Prompt text, included when the response is meant to hydrate the editor directly.",
    )


class LocalPromptArtifactListResponse(BaseModel):
    """Listing of local prompt artifacts for one prompt family."""

    purpose: Literal["character_description", "chat_translation"] = Field(
        ...,
        description="Prompt family included in the list.",
    )
    prompts: list[PromptArtifactSummary] = Field(
        default_factory=list,
        description="Selectable prompt files for the requested family.",
    )
    reference: ArtifactPromptReference = Field(
        ...,
        description="Reference contract to apply when editing prompts from this family.",
    )


class PromptArtifactDocument(BaseModel):
    """Full document payload returned when loading one prompt into the editor."""

    name: str = Field(..., description="Prompt stem without extension.")
    purpose: Literal["character_description", "chat_translation"] = Field(
        ...,
        description="Prompt family that owns the file.",
    )
    content: str = Field(..., description="Raw prompt text.")
    is_draft: bool = Field(..., description="True when the file is stored under drafts/.")
    origin_path: str = Field(..., description="Relative file path from the family root.")
    reference: ArtifactPromptReference = Field(
        ...,
        description="Reference contract to use while editing this prompt.",
    )


class ServerPromptManifestResponse(BaseModel):
    """Server-backed prompt artifact manifest for a specific world."""

    world_id: str = Field(..., description="Mud server world identifier.")
    world_name: str = Field(..., description="Human-readable world name.")
    prompts: list[PromptArtifactSummary] = Field(
        default_factory=list,
        description="Canonical world prompt files exposed by the mud server.",
    )
    active_prompt_name: str | None = Field(
        default=None,
        description="Stem of the active prompt template path for this world, if known.",
    )
    reference: ArtifactPromptReference = Field(
        ...,
        description="Canonical prompt contract for the selected world.",
    )


class ServerPromptArtifactListResponse(BaseModel):
    """Listing of mud-server prompt drafts for one selected world."""

    world_id: str = Field(..., description="Mud server world identifier.")
    prompts: list[PromptArtifactSummary] = Field(
        default_factory=list,
        description="Server-backed draft prompt files available for the selected world.",
    )
    reference: ArtifactPromptReference = Field(
        ...,
        description="Reference contract to apply when editing server-backed prompt drafts.",
    )


class LocalPromptDraftCreateRequest(BaseModel):
    """Create-only request for saving a new local draft prompt."""

    purpose: Literal["character_description", "chat_translation"] = Field(
        ...,
        description="Prompt family to store the draft under.",
    )
    draft_name: str = Field(
        ...,
        min_length=1,
        description="Filename stem for the new draft, without the .txt extension.",
    )
    content: str = Field(
        ...,
        description="Prompt text to write to the draft file.",
    )
    based_on_name: str | None = Field(
        default=None,
        description="Optional source prompt name this draft was derived from.",
    )


class LocalPromptDraftCreateResponse(BaseModel):
    """Response returned after creating a local draft prompt."""

    name: str = Field(..., description="Created draft stem without extension.")
    purpose: Literal["character_description", "chat_translation"] = Field(
        ...,
        description="Prompt family that now contains the draft.",
    )
    origin_path: str = Field(
        ...,
        description="Relative path of the newly created draft file.",
    )
    based_on_name: str | None = Field(
        default=None,
        description="Source prompt name copied into the request, if any.",
    )


class ServerPromptDraftCreateRequest(BaseModel):
    """Create-only request for saving a new mud-server prompt draft."""

    draft_name: str = Field(
        ...,
        min_length=1,
        description="Filename stem for the new server draft, without the .txt extension.",
    )
    content: str = Field(
        ...,
        description="Raw prompt text to forward to the mud server.",
    )
    based_on_name: str | None = Field(
        default=None,
        description="Optional source prompt name this draft was derived from.",
    )


class ServerPromptDraftCreateResponse(BaseModel):
    """Response returned after creating a new mud-server prompt draft."""

    name: str = Field(..., description="Created draft stem without extension.")
    origin_path: str = Field(
        ...,
        description="Canonical server-relative path of the newly created draft file.",
    )
    world_id: str = Field(..., description="World ID that owns the saved server draft.")
    based_on_name: str | None = Field(
        default=None,
        description="Source prompt name copied into the request, if any.",
    )


class AxisPayloadFieldInfo(BaseModel):
    """One documented field in the AxisPayload JSON contract."""

    name: str = Field(..., description="Top-level field name.")
    type: str = Field(..., description="Human-readable type description.")
    description: str = Field(..., description="Short explanation of the field's role.")


class AxisPayloadReference(BaseModel):
    """Reference metadata for Axis Payload JSON artifacts."""

    fields: list[AxisPayloadFieldInfo] = Field(
        default_factory=list,
        description="Documented top-level fields in the AxisPayload contract.",
    )
    sample_json: str = Field(
        ...,
        description="Canonical pretty-printed example JSON shown in the editor sidebar.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Constraints and invariants for AxisPayload JSON files.",
    )


class AxisPayloadArtifactSummary(BaseModel):
    """Metadata for one selectable AxisPayload JSON artifact."""

    name: str = Field(..., description="Artifact stem without the .json extension.")
    is_draft: bool = Field(
        ...,
        description="True when the file lives under a drafts/ directory rather than the shipped examples root.",
    )
    origin_path: str = Field(
        ...,
        description="Path relative to the examples root.",
    )
    world_id: str = Field(..., description="World ID declared by the payload.")


class LocalAxisPayloadArtifactListResponse(BaseModel):
    """Listing response for local AxisPayload JSON artifacts."""

    payloads: list[AxisPayloadArtifactSummary] = Field(
        default_factory=list,
        description="Selectable AxisPayload JSON files from the examples tree.",
    )
    reference: AxisPayloadReference = Field(
        ...,
        description="Reference contract for AxisPayload JSON editing.",
    )


class AxisPayloadArtifactDocument(BaseModel):
    """Full document payload for one AxisPayload JSON artifact."""

    name: str = Field(..., description="Artifact stem without extension.")
    content: str = Field(..., description="Pretty-printed raw JSON text.")
    is_draft: bool = Field(..., description="True when stored under examples/drafts/.")
    origin_path: str = Field(..., description="Path relative to the examples root.")
    world_id: str = Field(..., description="World ID declared by the payload.")
    reference: AxisPayloadReference = Field(
        ...,
        description="Reference contract to use while editing this payload.",
    )


class LocalAxisPayloadDraftCreateRequest(BaseModel):
    """Create-only request for saving a new local AxisPayload JSON draft."""

    draft_name: str = Field(
        ...,
        min_length=1,
        description="Filename stem for the new draft, without the .json extension.",
    )
    content: str = Field(
        ...,
        description="Raw JSON text to validate and save.",
    )
    based_on_name: str | None = Field(
        default=None,
        description="Optional source payload name this draft was derived from.",
    )


class LocalAxisPayloadDraftCreateResponse(BaseModel):
    """Response returned after creating a local AxisPayload JSON draft."""

    name: str = Field(..., description="Created draft stem without extension.")
    origin_path: str = Field(
        ...,
        description="Relative path of the newly created draft JSON file.",
    )
    world_id: str = Field(..., description="World ID declared by the saved payload.")
    based_on_name: str | None = Field(
        default=None,
        description="Source payload name copied into the request, if any.",
    )


class LexiconJsonFieldInfo(BaseModel):
    """One documented field in a deterministic lexicon JSON artifact."""

    name: str = Field(..., description="Top-level field name.")
    type: str = Field(..., description="Human-readable type description.")
    description: str = Field(..., description="Short explanation of the field's role.")


class LexiconJsonReference(BaseModel):
    """Reference metadata for deterministic lexicon JSON artifacts."""

    artifact_kind: Literal["catalog", "abstraction", "embodiment", "intensity"] = Field(
        ...,
        description="Which lexicon contract this reference describes.",
    )
    fields: list[LexiconJsonFieldInfo] = Field(
        default_factory=list,
        description="Documented top-level fields in the selected lexicon contract.",
    )
    sample_json: str = Field(
        ...,
        description="Canonical pretty-printed example JSON shown in the editor sidebar.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Constraints and invariants for deterministic lexicon JSON files.",
    )


class LexiconJsonArtifactSummary(BaseModel):
    """Metadata for one selectable deterministic lexicon JSON artifact."""

    name: str = Field(..., description="Artifact stem without the .json extension.")
    artifact_kind: Literal["abstraction", "embodiment", "intensity"] = Field(
        ...,
        description="Which deterministic lexicon contract this file follows.",
    )
    is_draft: bool = Field(
        ...,
        description="True when the file lives under a drafts/ directory rather than the shipped data root.",
    )
    origin_path: str = Field(
        ...,
        description="Path relative to the app/data root.",
    )
    version: str = Field(..., description="Version declared by the lexicon file.")


class LocalLexiconJsonArtifactListResponse(BaseModel):
    """Listing response for local deterministic lexicon JSON artifacts."""

    lexicons: list[LexiconJsonArtifactSummary] = Field(
        default_factory=list,
        description="Selectable lexicon JSON files from the app/data tree.",
    )
    reference: LexiconJsonReference = Field(
        ...,
        description="Reference contract for lexicon JSON editing.",
    )


class LexiconJsonArtifactDocument(BaseModel):
    """Full document payload for one deterministic lexicon JSON artifact."""

    name: str = Field(..., description="Artifact stem without extension.")
    artifact_kind: Literal["abstraction", "embodiment", "intensity"] = Field(
        ...,
        description="Which deterministic lexicon contract this file follows.",
    )
    content: str = Field(..., description="Pretty-printed raw JSON text.")
    is_draft: bool = Field(..., description="True when stored under data/drafts/.")
    origin_path: str = Field(..., description="Path relative to the data root.")
    version: str = Field(..., description="Version declared by the lexicon file.")
    reference: LexiconJsonReference = Field(
        ...,
        description="Reference contract to use while editing this lexicon.",
    )


class LocalLexiconJsonDraftCreateRequest(BaseModel):
    """Create-only request for saving a new local deterministic lexicon JSON draft."""

    draft_name: str = Field(
        ...,
        min_length=1,
        description="Filename stem for the new draft, without the .json extension.",
    )
    content: str = Field(
        ...,
        description="Raw JSON text to validate and save.",
    )
    based_on_name: str | None = Field(
        default=None,
        description="Optional source lexicon name this draft was derived from.",
    )


class LocalLexiconJsonDraftCreateResponse(BaseModel):
    """Response returned after creating a local deterministic lexicon JSON draft."""

    name: str = Field(..., description="Created draft stem without extension.")
    artifact_kind: Literal["abstraction", "embodiment", "intensity"] = Field(
        ...,
        description="Which deterministic lexicon contract the saved file follows.",
    )
    origin_path: str = Field(
        ...,
        description="Relative path of the newly created draft JSON file.",
    )
    version: str = Field(..., description="Version declared by the saved lexicon.")
    based_on_name: str | None = Field(
        default=None,
        description="Source lexicon name copied into the request, if any.",
    )


class PolicyBundleFieldInfo(BaseModel):
    """One documented field in a normalized world policy bundle JSON artifact."""

    name: str = Field(..., description="Top-level field name.")
    type: str = Field(..., description="Human-readable type description.")
    description: str = Field(..., description="Short explanation of the field's role.")


class PolicyBundleReference(BaseModel):
    """Reference metadata for normalized world policy bundle JSON artifacts."""

    fields: list[PolicyBundleFieldInfo] = Field(
        default_factory=list,
        description="Documented top-level fields in the policy bundle contract.",
    )
    sample_json: str = Field(
        ...,
        description="Canonical pretty-printed example JSON shown in the editor sidebar.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Constraints and invariants for policy bundle JSON files.",
    )


class PolicyBundleArtifactSummary(BaseModel):
    """Metadata for one selectable normalized world policy bundle JSON artifact."""

    name: str = Field(..., description="Artifact stem without the .json extension.")
    is_draft: bool = Field(
        ...,
        description="True when the file lives under a drafts/ directory rather than the shipped policy bundle root.",
    )
    origin_path: str = Field(
        ...,
        description="Path relative to the policy bundle root.",
    )
    world_id: str = Field(..., description="World identifier declared by the policy bundle.")
    version: str = Field(..., description="Version declared by the policy bundle.")


class LocalPolicyBundleArtifactListResponse(BaseModel):
    """Listing response for local normalized world policy bundle JSON artifacts."""

    bundles: list[PolicyBundleArtifactSummary] = Field(
        default_factory=list,
        description="Selectable policy bundle JSON files from the policy bundle tree.",
    )
    reference: PolicyBundleReference = Field(
        ...,
        description="Reference contract for policy bundle JSON editing.",
    )


class PolicyBundleArtifactDocument(BaseModel):
    """Full document payload for one normalized world policy bundle JSON artifact."""

    name: str = Field(..., description="Artifact stem without extension.")
    content: str = Field(..., description="Pretty-printed raw JSON text.")
    is_draft: bool = Field(..., description="True when stored under policy bundle drafts/.")
    origin_path: str = Field(..., description="Path relative to the policy bundle root.")
    world_id: str = Field(..., description="World identifier declared by the policy bundle.")
    version: str = Field(..., description="Version declared by the policy bundle.")
    reference: PolicyBundleReference = Field(
        ...,
        description="Reference contract to use while editing this policy bundle.",
    )


class LocalPolicyBundleDraftCreateRequest(BaseModel):
    """Create-only request for saving a new local normalized policy bundle JSON draft."""

    draft_name: str = Field(
        ...,
        min_length=1,
        description="Filename stem for the new draft, without the .json extension.",
    )
    content: str = Field(
        ...,
        description="Raw JSON text to validate and save.",
    )
    based_on_name: str | None = Field(
        default=None,
        description="Optional source policy bundle name this draft was derived from.",
    )


class LocalPolicyBundleDraftCreateResponse(BaseModel):
    """Response returned after creating a local normalized policy bundle JSON draft."""

    name: str = Field(..., description="Created draft stem without extension.")
    origin_path: str = Field(
        ...,
        description="Relative path of the newly created draft JSON file.",
    )
    world_id: str = Field(..., description="World identifier declared by the saved policy bundle.")
    version: str = Field(..., description="Version declared by the saved policy bundle.")
    based_on_name: str | None = Field(
        default=None,
        description="Source policy bundle name copied into the request, if any.",
    )


class ServerPolicyBundleDraftCreateRequest(BaseModel):
    """Create-only request for saving a new mud-server policy bundle draft."""

    draft_name: str = Field(
        ...,
        min_length=1,
        description="Filename stem for the new server draft, without the .json extension.",
    )
    content: str = Field(
        ...,
        description="Raw normalized policy bundle JSON text to validate and forward.",
    )
    based_on_name: str | None = Field(
        default=None,
        description="Optional source bundle name this draft was derived from.",
    )


class ServerPolicyBundleDraftCreateResponse(BaseModel):
    """Response returned after creating a new mud-server policy bundle draft."""

    name: str = Field(..., description="Created draft stem without extension.")
    origin_path: str = Field(
        ...,
        description="Canonical server-relative path of the newly created draft file.",
    )
    world_id: str = Field(..., description="World ID that owns the saved server draft.")
    version: str = Field(..., description="Version declared by the saved policy bundle.")
    based_on_name: str | None = Field(
        default=None,
        description="Source bundle name copied into the request, if any.",
    )


class ServerPolicyBundleArtifactListResponse(BaseModel):
    """Listing of mud-server policy bundle drafts for one selected world."""

    world_id: str = Field(..., description="Mud server world identifier.")
    bundles: list[PolicyBundleArtifactSummary] = Field(
        default_factory=list,
        description="Server-backed draft bundle files available for the selected world.",
    )
    reference: PolicyBundleReference = Field(
        ...,
        description="Reference contract to apply when editing server-backed policy bundles.",
    )
