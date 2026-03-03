"""
Prompt-artifact services for the Artifact Editor page.

The planned Artifact Editor will eventually cover more than prompt files, but
the first implementation focuses on prompt-template editing because that is
where drift between the lab and the mud server is currently the most visible.

Responsibilities in this module:

- list and load local prompt artifacts, including shipped files and local
  drafts under ``app/prompts/*/drafts``
- create new local draft prompt files without overwriting any existing file
- derive prompt reference metadata for the editor sidebar
- normalise server-backed prompt manifests from the mud-server proxy client

This module intentionally does not write to the mud server.  Server-backed
editing is read-only in the first cut so the mud server remains the canonical
source of truth while the lab provides the editing UX.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from fastapi import HTTPException

from app.file_loaders import PROMPTS_DIR, load_prompt
from app.mud_server_client import MudServerClient
from app.schema.artifact import (
    ArtifactPlaceholder,
    ArtifactPromptReference,
    LocalPromptArtifactListResponse,
    LocalPromptDraftCreateRequest,
    LocalPromptDraftCreateResponse,
    PromptArtifactDocument,
    PromptArtifactSummary,
    ServerPromptManifestResponse,
)

type PromptPurpose = Literal["character_description", "chat_translation"]

_DRAFT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _prompt_root(purpose: PromptPurpose) -> Path:
    """Return the filesystem root for one prompt family."""

    return PROMPTS_DIR / purpose


def _iter_prompt_files(purpose: PromptPurpose) -> list[Path]:
    """Return all prompt files, including drafts, for one prompt family."""

    return sorted(_prompt_root(purpose).rglob("*.txt"))


def _is_draft(path: Path, purpose: PromptPurpose) -> bool:
    """Return True when the file lives inside the family's drafts directory."""

    return "drafts" in path.relative_to(_prompt_root(purpose)).parts


def _relative_origin_path(path: Path, purpose: PromptPurpose) -> str:
    """Return a prompt file path relative to its family root."""

    return path.relative_to(_prompt_root(purpose)).as_posix()


def _build_profile_summary_example(active_axes: list[str]) -> str:
    """Build a mud-server-style example ``profile_summary`` block.

    The example mirrors the formatting used by the mud server's translation
    layer so the Artifact Editor can show users what ``{{profile_summary}}``
    actually looks like when a canonical world prompt is rendered.
    """

    if not active_axes:
        active_axes = ["demeanor", "health", "physique"]

    lines = ["  Character: Example Character"]
    for index, axis_name in enumerate(active_axes, start=1):
        display_name = axis_name.replace("_", " ").title()
        lines.append(f"  {display_name}: example_{axis_name} ({index / 10:.2f})")
    return "\n".join(lines)


def _chat_translation_reference(
    *, source_mode: Literal["local", "server"], world_id: str | None, active_axes: list[str]
) -> ArtifactPromptReference:
    """Return the prompt contract for chat-translation templates."""

    placeholders = [
        ArtifactPlaceholder(
            placeholder="{{profile_summary}}",
            description="Canonical multi-line character summary rendered by the translation pipeline.",
        ),
        ArtifactPlaceholder(
            placeholder="{{channel}}",
            description="Delivery mode such as say, yell, or whisper.",
        ),
        ArtifactPlaceholder(
            placeholder="{{ooc_message}}",
            description="Raw out-of-character text when a prompt embeds it inside the system prompt.",
        ),
        ArtifactPlaceholder(
            placeholder="{{character_name}}",
            description="Character display name when a prompt addresses it directly.",
        ),
    ]
    sample_values = {
        "profile_summary": _build_profile_summary_example(active_axes),
        "channel": "say",
        "ooc_message": "Can you help me with this ledger?",
        "character_name": "Example Character",
    }

    for index, axis_name in enumerate(active_axes, start=1):
        label_key = f"{axis_name}_label"
        score_key = f"{axis_name}_score"
        placeholders.extend(
            [
                ArtifactPlaceholder(
                    placeholder=f"{{{{{label_key}}}}}",
                    description=f"Resolved label for the '{axis_name}' axis.",
                ),
                ArtifactPlaceholder(
                    placeholder=f"{{{{{score_key}}}}}",
                    description=f"Resolved numeric score for the '{axis_name}' axis.",
                ),
            ]
        )
        sample_values[label_key] = f"example_{axis_name}"
        sample_values[score_key] = f"{index / 10:.2f}"

    notes = [
        "The chat-translation pipeline sends the OOC message as the user turn even when the prompt also embeds {{ooc_message}}.",
        "profile_summary formatting should match the mud server's canonical translation service.",
    ]
    if source_mode == "local":
        notes.append(
            "Local mode validates against the lab's current standalone chat contract; server-backed mode is authoritative for world-specific active axes."
        )

    return ArtifactPromptReference(
        source_mode=source_mode,
        purpose="chat_translation",
        world_id=world_id,
        active_axes=active_axes,
        placeholders=placeholders,
        sample_values=sample_values,
        profile_summary_example=sample_values["profile_summary"],
        notes=notes,
    )


def _character_description_reference() -> ArtifactPromptReference:
    """Return the prompt contract for Character Description prompts."""

    return ArtifactPromptReference(
        source_mode="local",
        purpose="character_description",
        world_id=None,
        active_axes=[],
        placeholders=[],
        sample_values={},
        profile_summary_example=None,
        notes=[
            "Character Description prompts receive the deterministic AxisPayload JSON as the user turn rather than template placeholders inside the system prompt.",
            "This prompt family remains a standalone lab artifact and is not sourced from the mud server.",
        ],
    )


def build_prompt_reference(
    purpose: PromptPurpose,
    *,
    source_mode: Literal["local", "server"] = "local",
    world_id: str | None = None,
    active_axes: list[str] | None = None,
) -> ArtifactPromptReference:
    """Build the editor reference contract for one prompt family."""

    if purpose == "character_description":
        return _character_description_reference()
    return _chat_translation_reference(
        source_mode=source_mode,
        world_id=world_id,
        active_axes=list(active_axes or []),
    )


def list_local_prompt_artifacts(purpose: PromptPurpose) -> LocalPromptArtifactListResponse:
    """List shipped and draft prompt files for one local prompt family."""

    prompts = [
        PromptArtifactSummary(
            name=path.stem,
            purpose=purpose,
            is_draft=_is_draft(path, purpose),
            is_active=False,
            origin_path=_relative_origin_path(path, purpose),
        )
        for path in _iter_prompt_files(purpose)
    ]
    return LocalPromptArtifactListResponse(
        purpose=purpose,
        prompts=prompts,
        reference=build_prompt_reference(purpose),
    )


def load_local_prompt_artifact(name: str, purpose: PromptPurpose) -> PromptArtifactDocument:
    """Load one local prompt file together with its editor contract."""

    target: Path | None = None
    for path in _iter_prompt_files(purpose):
        if path.stem == name:
            target = path
            break

    if target is None:
        raise HTTPException(status_code=404, detail=f"Prompt artifact '{name}' not found.")

    return PromptArtifactDocument(
        name=name,
        purpose=purpose,
        content=load_prompt(name, purpose=purpose),
        is_draft=_is_draft(target, purpose),
        origin_path=_relative_origin_path(target, purpose),
        reference=build_prompt_reference(purpose),
    )


def create_local_prompt_draft(req: LocalPromptDraftCreateRequest) -> LocalPromptDraftCreateResponse:
    """Create a new prompt draft under ``app/prompts/<purpose>/drafts``.

    Overwriting any existing filename is forbidden, including collisions with
    shipped prompt files.  This keeps the editor on the safe side of the
    "create drafts, do not overwrite canonical artifacts" rule.
    """

    draft_name = req.draft_name.strip()
    if not _DRAFT_NAME_RE.fullmatch(draft_name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Draft names must use lowercase letters, numbers, underscores, or hyphens "
                "and must not include a file extension."
            ),
        )

    purpose_root = _prompt_root(req.purpose)
    existing_names = {path.stem for path in _iter_prompt_files(req.purpose)}
    if draft_name in existing_names:
        raise HTTPException(
            status_code=409,
            detail=f"A prompt named '{draft_name}' already exists in {req.purpose}.",
        )

    drafts_dir = purpose_root / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    target = drafts_dir / f"{draft_name}.txt"
    target.write_text(req.content.rstrip() + "\n", encoding="utf-8")

    return LocalPromptDraftCreateResponse(
        name=draft_name,
        purpose=req.purpose,
        origin_path=_relative_origin_path(target, req.purpose),
        based_on_name=req.based_on_name,
    )


def get_server_prompt_manifest(
    world_id: str,
    mud_client: MudServerClient,
) -> ServerPromptManifestResponse:
    """Build a normalized prompt manifest from the mud server's lab endpoints."""

    world_cfg = mud_client.world_config(world_id)
    prompt_data = mud_client.world_prompts(world_id)
    prompts_raw = prompt_data.get("prompts", [])
    active_axes = list(world_cfg.get("active_axes") or [])

    prompts: list[PromptArtifactSummary] = []
    active_prompt_name: str | None = None
    for entry in prompts_raw:
        filename = str(entry.get("filename", ""))
        stem = Path(filename).stem
        is_active = bool(entry.get("is_active"))
        if is_active:
            active_prompt_name = stem
        prompts.append(
            PromptArtifactSummary(
                name=stem,
                purpose="chat_translation",
                is_draft=False,
                is_active=is_active,
                origin_path=filename,
                content=str(entry.get("content", "")),
            )
        )

    world_name = str(world_cfg.get("name") or world_id)
    return ServerPromptManifestResponse(
        world_id=world_id,
        world_name=world_name,
        prompts=prompts,
        active_prompt_name=active_prompt_name,
        reference=build_prompt_reference(
            "chat_translation",
            source_mode="server",
            world_id=world_id,
            active_axes=active_axes,
        ),
    )
