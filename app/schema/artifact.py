"""
Schemas for the Artifact Editor page.

These models describe the prompt-artifact subset of the planned Artifact
Editor workflow.  The first implementation focuses on prompt templates only:

- local prompt files stored in ``app/prompts/*`` plus draft files under
  ``app/prompts/*/drafts``
- server-backed prompt manifests derived from the mud server's canonical world
  prompt endpoints
- create-only local draft saves that avoid overwriting shipped prompt files

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
