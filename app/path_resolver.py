"""
Deterministic path resolution for supported Axis Lab local artifacts.

Axis Lab now supports only two local asset tiers:

1. world-scoped roots
2. explicitly lab-only roots

High-level precedence remains:

1. world-scoped roots (canonical and draft)
2. lab-only roots
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.config import (
    DEFAULT_WORLD_ID,
    LAB_ONLY_ROOT,
    WORLD_ROOT,
)

type ArtifactSourceKind = Literal["world_canonical", "world_draft", "lab_only"]
type PromptPurpose = Literal["character_description", "chat_translation"]


class PathResolutionError(RuntimeError):
    """Raised when artifact resolution is ambiguous at the same precedence level."""


@dataclass(frozen=True)
class ResolvedArtifactPath:
    """Resolved artifact file and its source metadata.

    Attributes:
        stem: File stem without extension (lookup key used by current APIs).
        path: Absolute filesystem path to the resolved file.
        source_kind: Source bucket that produced this file.
        source_path: Path relative to the source root.
        world_id: Owning world id for world-scoped artifacts, else ``None``.
        is_draft: ``True`` when the file belongs to a draft subtree.
        priority: Internal precedence rank; lower values win.
    """

    stem: str
    path: Path
    source_kind: ArtifactSourceKind
    source_path: str
    world_id: str | None
    is_draft: bool
    priority: int


def _scan_files(
    *,
    root: Path,
    pattern: str,
    source_kind: ArtifactSourceKind,
    priority: int,
    world_id: str | None = None,
    force_draft: bool | None = None,
    source_prefix: str | None = None,
) -> list[ResolvedArtifactPath]:
    """Collect resolved artifact metadata from one root and glob pattern."""

    if not root.exists():
        return []

    results: list[ResolvedArtifactPath] = []
    for path in sorted(root.rglob(pattern)):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        source_rel = rel.as_posix()
        if source_prefix:
            source_rel = f"{source_prefix.rstrip('/')}/{source_rel}"
        is_draft = force_draft if force_draft is not None else "drafts" in rel.parts
        results.append(
            ResolvedArtifactPath(
                stem=path.stem,
                path=path,
                source_kind=source_kind,
                source_path=source_rel,
                world_id=world_id,
                is_draft=is_draft,
                priority=priority,
            )
        )
    return results


def _build_stem_index(
    candidates: list[ResolvedArtifactPath],
    *,
    context: str,
) -> dict[str, ResolvedArtifactPath]:
    """Resolve candidates to one deterministic entry per stem.

    If two candidates with the same stem exist at the same precedence level,
    resolution is ambiguous and the caller receives ``PathResolutionError``.
    """

    index: dict[str, ResolvedArtifactPath] = {}
    for candidate in sorted(candidates, key=lambda row: (row.priority, row.stem, row.source_path)):
        existing = index.get(candidate.stem)
        if existing is None:
            index[candidate.stem] = candidate
            continue
        if existing.priority == candidate.priority:
            raise PathResolutionError(
                f"Duplicate {context} name '{candidate.stem}' at precedence {candidate.priority}: "
                f"{existing.path} and {candidate.path}"
            )
    return index


def sorted_resolved_paths(index: dict[str, ResolvedArtifactPath]) -> list[ResolvedArtifactPath]:
    """Return resolved rows sorted by stem for stable list rendering."""

    return sorted(index.values(), key=lambda row: row.stem)


def resolve_prompt_paths(
    purpose: PromptPurpose,
    *,
    world_id: str = DEFAULT_WORLD_ID,
    world_root: Path = WORLD_ROOT,
    lab_only_root: Path = LAB_ONLY_ROOT,
) -> dict[str, ResolvedArtifactPath]:
    """Resolve prompt files by stem with deterministic world/lab precedence."""

    candidates: list[ResolvedArtifactPath] = []

    if purpose == "chat_translation":
        canonical_root = world_root / world_id / "policies" / "translation" / "prompts" / "ic"
        candidates.extend(
            _scan_files(
                root=canonical_root,
                pattern="*.txt",
                source_kind="world_canonical",
                priority=0,
                world_id=world_id,
                force_draft=False,
                source_prefix="policies/translation/prompts/ic",
            )
        )
        draft_root = (
            world_root / world_id / "policies" / "drafts" / "translation" / "prompts" / "ic"
        )
        candidates.extend(
            _scan_files(
                root=draft_root,
                pattern="*.txt",
                source_kind="world_draft",
                priority=1,
                world_id=world_id,
                force_draft=True,
                source_prefix="policies/drafts/translation/prompts/ic",
            )
        )
        candidates.extend(
            _scan_files(
                root=lab_only_root / "prompts" / "chat_translation",
                pattern="*.txt",
                source_kind="lab_only",
                priority=2,
            )
        )
        return _build_stem_index(candidates, context="chat prompt")

    candidates.extend(
        _scan_files(
            root=lab_only_root / "prompts" / "character_description",
            pattern="*.txt",
            source_kind="lab_only",
            priority=0,
        )
    )
    return _build_stem_index(candidates, context="character prompt")


def resolve_axis_payload_paths(
    *,
    world_id: str = DEFAULT_WORLD_ID,
    world_root: Path = WORLD_ROOT,
    lab_only_root: Path = LAB_ONLY_ROOT,
) -> dict[str, ResolvedArtifactPath]:
    """Resolve AxisPayload examples by stem with deterministic precedence."""

    candidates: list[ResolvedArtifactPath] = []
    candidates.extend(
        _scan_files(
            root=world_root / world_id / "policies" / "axis" / "examples",
            pattern="*.json",
            source_kind="world_canonical",
            priority=0,
            world_id=world_id,
            force_draft=False,
            source_prefix="policies/axis/examples",
        )
    )
    candidates.extend(
        _scan_files(
            root=world_root / world_id / "policies" / "drafts" / "axis" / "examples",
            pattern="*.json",
            source_kind="world_draft",
            priority=1,
            world_id=world_id,
            force_draft=True,
            source_prefix="policies/drafts/axis/examples",
        )
    )
    candidates.extend(
        _scan_files(
            root=lab_only_root / "axis" / "examples",
            pattern="*.json",
            source_kind="lab_only",
            priority=2,
        )
    )
    return _build_stem_index(candidates, context="axis payload")


def resolve_lexicon_paths(
    *,
    lab_only_root: Path = LAB_ONLY_ROOT,
) -> dict[str, ResolvedArtifactPath]:
    """Resolve deterministic lexicon JSON files by stem with lab-only precedence."""

    candidates: list[ResolvedArtifactPath] = []
    candidates.extend(
        _scan_files(
            root=lab_only_root / "axis" / "lexicons",
            pattern="*.json",
            source_kind="lab_only",
            priority=0,
        )
    )
    return _build_stem_index(candidates, context="lexicon")


def resolve_policy_bundle_paths(
    *,
    world_id: str = DEFAULT_WORLD_ID,
    world_root: Path = WORLD_ROOT,
    lab_only_root: Path = LAB_ONLY_ROOT,
) -> dict[str, ResolvedArtifactPath]:
    """Resolve normalized policy bundle JSON files with deterministic precedence."""

    candidates: list[ResolvedArtifactPath] = []
    candidates.extend(
        _scan_files(
            root=world_root / world_id / "policies" / "drafts" / "policy_bundles",
            pattern="*.json",
            source_kind="world_draft",
            priority=0,
            world_id=world_id,
            force_draft=True,
            source_prefix="policies/drafts/policy_bundles",
        )
    )
    candidates.extend(
        _scan_files(
            root=lab_only_root / "policy_bundles",
            pattern="*.json",
            source_kind="lab_only",
            priority=1,
        )
    )
    return _build_stem_index(candidates, context="policy bundle")
