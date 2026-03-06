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
- create, list, and load mud-server prompt drafts without overwriting
  canonical server files

This module intentionally keeps the mud server authoritative.  Server-backed
editing loads canonical artifacts from the mud server and may create new
draft files there, but it never overwrites active canonical files.

The local-only JSON slices extend that same pattern to deterministic artifacts
owned entirely by the lab, including:

- AxisPayload JSON examples under ``app/examples``
- normalized world policy bundle JSON files under ``app/artifacts/policy_bundles``
- micro-indicator lexicon/config JSON files under ``app/data``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from app.config import (
    DEFAULT_WORLD_ID,
    LAB_ONLY_ROOT,
    WORLD_ROOT,
)
from app.file_loaders import EXAMPLES_DIR, PROMPTS_DIR
from app.mud_server_client import MudServerClient
from app.path_resolver import (
    PathResolutionError,
    ResolvedArtifactPath,
    resolve_axis_payload_paths,
    resolve_lexicon_paths,
    resolve_policy_bundle_paths,
    resolve_prompt_paths,
    sorted_resolved_paths,
)
from app.schema.axis import AxisPayload
from app.schema.artifact import (
    AxisPayloadArtifactDocument,
    AxisPayloadArtifactSummary,
    AxisPayloadFieldInfo,
    AxisPayloadReference,
    ArtifactPlaceholder,
    ArtifactPromptReference,
    LocalAxisPayloadArtifactListResponse,
    LocalAxisPayloadDraftCreateRequest,
    LocalAxisPayloadDraftCreateResponse,
    LexiconJsonArtifactDocument,
    LexiconJsonArtifactSummary,
    LexiconJsonFieldInfo,
    LexiconJsonReference,
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
    PolicyBundleArtifactSummary,
    PolicyBundleFieldInfo,
    PolicyBundleReference,
    PromptArtifactDocument,
    PromptArtifactSummary,
    ServerPromptArtifactListResponse,
    ServerPromptDraftCreateRequest,
    ServerPromptDraftCreateResponse,
    ServerPromptDraftPromoteRequest,
    ServerPromptDraftPromoteResponse,
    ServerPolicyBundleDraftCreateRequest,
    ServerPolicyBundleArtifactListResponse,
    ServerPolicyBundleDraftCreateResponse,
    ServerPolicyBundleDraftPromoteRequest,
    ServerPolicyBundleDraftPromoteResponse,
    ServerPromptManifestResponse,
)

type PromptPurpose = Literal["character_description", "chat_translation"]
type LexiconKind = Literal["abstraction", "embodiment", "intensity"]

_DRAFT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
DATA_DIR = Path(__file__).parent / "data"
POLICY_BUNDLES_DIR = Path(__file__).parent / "artifacts" / "policy_bundles"
WORLD_ASSET_ROOT = WORLD_ROOT
LAB_ONLY_ASSET_ROOT = LAB_ONLY_ROOT
DEFAULT_ASSET_WORLD_ID = DEFAULT_WORLD_ID


class AbstractionLexiconPayload(BaseModel):
    """Validated structure for abstraction/concrete lexicon JSON files."""

    model_config = ConfigDict(extra="forbid")

    version: str
    abstract_terms: list[str]
    concrete_terms: list[str]


class EmbodimentLexiconPayload(BaseModel):
    """Validated structure for embodiment lexicon JSON files."""

    model_config = ConfigDict(extra="forbid")

    version: str
    abstract: list[str]
    physical: list[str]


class IntensityLexiconPayload(BaseModel):
    """Validated structure for intensity-scale lexicon JSON files."""

    model_config = ConfigDict(extra="forbid")

    version: str
    scales: dict[str, list[str]]


class PolicyThresholdRange(BaseModel):
    """One ordinal threshold band in a normalized world policy bundle."""

    label: str
    min: float
    max: float


class PolicyAxisDefinition(BaseModel):
    """Normalized per-axis metadata derived from mud-server policy files."""

    model_config = ConfigDict(extra="forbid")

    group: str
    ordering: list[str]
    thresholds: list[PolicyThresholdRange]

    @model_validator(mode="after")
    def validate_threshold_ordering(self) -> "PolicyAxisDefinition":
        """Ensure threshold labels align exactly with the declared ordering."""

        labels = [entry.label for entry in self.thresholds]
        if labels != self.ordering:
            raise ValueError("threshold labels must match ordering exactly")
        return self


class PolicyChatAxisRule(BaseModel):
    """Chat-resolution rule for one axis in a policy bundle."""

    model_config = ConfigDict(extra="forbid")

    resolver: str
    base_magnitude: float | None = None

    @model_validator(mode="after")
    def validate_rule(self) -> "PolicyChatAxisRule":
        """Require base magnitude on rules that are not explicit no-ops."""

        if self.resolver != "no_effect" and self.base_magnitude is None:
            raise ValueError("base_magnitude is required unless resolver is 'no_effect'")
        return self


class PolicyChatRules(BaseModel):
    """Normalized chat-resolution rules derived from resolution.yaml."""

    model_config = ConfigDict(extra="forbid")

    channel_multipliers: dict[str, float]
    min_gap_threshold: float
    axes: dict[str, PolicyChatAxisRule]

    @model_validator(mode="after")
    def validate_channels(self) -> "PolicyChatRules":
        """Require the canonical say/yell/whisper channel keys."""

        if set(self.channel_multipliers) != {"say", "yell", "whisper"}:
            raise ValueError("channel_multipliers must define exactly say, yell, and whisper")
        return self


class PolicyBundlePayload(BaseModel):
    """Local JSON normalization of a mud-server world policy package."""

    model_config = ConfigDict(extra="forbid")

    world_id: str
    version: str
    source: str
    policy_hash: str | None = None
    axes_order: list[str]
    axes: dict[str, PolicyAxisDefinition]
    chat_rules: PolicyChatRules

    @model_validator(mode="after")
    def validate_consistency(self) -> "PolicyBundlePayload":
        """Enforce consistent axis coverage across the normalized bundle."""

        if self.axes_order != list(self.axes.keys()):
            raise ValueError("axes_order must match the axes object key order exactly")
        if set(self.chat_rules.axes) != set(self.axes):
            raise ValueError("chat_rules.axes must define exactly the same axis set as axes")
        return self


def _prompt_root(purpose: PromptPurpose) -> Path:
    """Return the filesystem root for one prompt family."""

    return PROMPTS_DIR / purpose


def _examples_root() -> Path:
    """Return the filesystem root for AxisPayload example files."""

    return EXAMPLES_DIR


def _data_root() -> Path:
    """Return the filesystem root for deterministic lexicon JSON files."""

    return DATA_DIR


def _policy_bundle_root() -> Path:
    """Return the filesystem root for normalized local policy bundle JSON files."""

    return POLICY_BUNDLES_DIR


def _iter_prompt_files(purpose: PromptPurpose) -> list[Path]:
    """Return resolved prompt file paths for one prompt family."""

    return [row.path for row in _resolved_prompt_rows(purpose)]


def _resolved_prompt_rows(purpose: PromptPurpose) -> list[ResolvedArtifactPath]:
    """Return resolved prompt rows with world->lab->legacy precedence."""

    try:
        index = resolve_prompt_paths(
            purpose,
            world_id=DEFAULT_ASSET_WORLD_ID,
            world_root=WORLD_ASSET_ROOT,
            lab_only_root=LAB_ONLY_ASSET_ROOT,
            legacy_prompts_root=PROMPTS_DIR,
        )
    except PathResolutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return sorted_resolved_paths(index)


def _relative_origin_path(path: Path, purpose: PromptPurpose) -> str:
    """Return a prompt file path relative to its family root."""

    return path.relative_to(_prompt_root(purpose)).as_posix()


def _iter_axis_payload_files() -> list[Path]:
    """Return resolved AxisPayload JSON files with world->lab->legacy precedence."""

    return [row.path for row in _resolved_axis_payload_rows()]


def _resolved_axis_payload_rows() -> list[ResolvedArtifactPath]:
    """Return resolved AxisPayload rows with source metadata."""

    try:
        index = resolve_axis_payload_paths(
            world_id=DEFAULT_ASSET_WORLD_ID,
            world_root=WORLD_ASSET_ROOT,
            lab_only_root=LAB_ONLY_ASSET_ROOT,
            legacy_examples_root=EXAMPLES_DIR,
        )
    except PathResolutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return sorted_resolved_paths(index)


def _axis_payload_origin_path(path: Path) -> str:
    """Return a payload path relative to the examples root."""

    return path.relative_to(_examples_root()).as_posix()


def _iter_lexicon_json_files() -> list[Path]:
    """Return resolved deterministic lexicon JSON files with lab->legacy precedence."""

    return [row.path for row in _resolved_lexicon_rows()]


def _resolved_lexicon_rows() -> list[ResolvedArtifactPath]:
    """Return resolved lexicon rows with source metadata."""

    try:
        index = resolve_lexicon_paths(
            lab_only_root=LAB_ONLY_ASSET_ROOT,
            legacy_lexicons_root=DATA_DIR,
        )
    except PathResolutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return sorted_resolved_paths(index)


def _lexicon_origin_path(path: Path) -> str:
    """Return a lexicon path relative to the data root."""

    return path.relative_to(_data_root()).as_posix()


def _iter_policy_bundle_files() -> list[Path]:
    """Return resolved policy bundle JSON files with draft-first precedence."""

    return [row.path for row in _resolved_policy_bundle_rows()]


def _resolved_policy_bundle_rows() -> list[ResolvedArtifactPath]:
    """Return resolved policy-bundle rows with source metadata."""

    try:
        index = resolve_policy_bundle_paths(
            world_id=DEFAULT_ASSET_WORLD_ID,
            world_root=WORLD_ASSET_ROOT,
            lab_only_root=LAB_ONLY_ASSET_ROOT,
            legacy_policy_bundle_root=POLICY_BUNDLES_DIR,
        )
    except PathResolutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return sorted_resolved_paths(index)


def _policy_bundle_origin_path(path: Path) -> str:
    """Return a policy bundle path relative to the policy bundle root."""

    return path.relative_to(_policy_bundle_root()).as_posix()


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


def _axis_payload_reference() -> AxisPayloadReference:
    """Return the schema/reference metadata for AxisPayload JSON artifacts."""

    sample_payload = {
        "axes": {
            "demeanor": {"label": "proud", "score": 0.81},
            "health": {"label": "weary", "score": 0.34},
            "wealth": {"label": "destitute", "score": 0.12},
        },
        "policy_hash": "example_policy_hash",
        "seed": 42,
        "world_id": "pipeworks_web",
    }
    return AxisPayloadReference(
        fields=[
            AxisPayloadFieldInfo(
                name="axes",
                type="object",
                description="Map of axis name to {label, score} entries. At least one axis is required.",
            ),
            AxisPayloadFieldInfo(
                name="policy_hash",
                type="string",
                description="Digest of the policy rules in force when the payload was produced.",
            ),
            AxisPayloadFieldInfo(
                name="seed",
                type="integer",
                description="Deterministic seed used to produce the payload.",
            ),
            AxisPayloadFieldInfo(
                name="world_id",
                type="string",
                description="Pipe-Works world identifier that scopes the payload.",
            ),
        ],
        sample_json=json.dumps(sample_payload, ensure_ascii=False, indent=2),
        notes=[
            "AxisPayload JSON is authoritative input data, not derived prompt text.",
            "Draft saves are validated against the AxisPayload schema before they are written.",
            "Local drafts are stored under app/examples/drafts and never overwrite shipped examples.",
        ],
    )


def _lexicon_catalog_reference() -> LexiconJsonReference:
    """Return generic reference metadata for the lexicon artifact catalog."""

    sample_catalog = {
        "abstraction_v0_1": "abstract_terms + concrete_terms",
        "embodiment_v0_1": "abstract + physical",
        "intensity_v0_1": "scales[name] = ordered words",
    }
    return LexiconJsonReference(
        artifact_kind="catalog",
        fields=[
            LexiconJsonFieldInfo(
                name="version",
                type="string",
                description="Semantic data version embedded in the file.",
            ),
            LexiconJsonFieldInfo(
                name="contract-specific fields",
                type="object",
                description="Each lexicon kind has a fixed top-level shape validated before draft save.",
            ),
        ],
        sample_json=json.dumps(sample_catalog, ensure_ascii=False, indent=2),
        notes=[
            "These JSON files drive deterministic micro-indicator heuristics in app/micro_indicators.py.",
            "Draft saves are validated against one supported lexicon contract before they are written.",
            "Local drafts are stored under app/data/drafts and never overwrite shipped lexicon files.",
        ],
    )


def _lexicon_reference(kind: LexiconKind) -> LexiconJsonReference:
    """Return schema/reference metadata for one deterministic lexicon contract."""

    if kind == "abstraction":
        sample_payload: dict[str, object] = {
            "version": "0.1",
            "abstract_terms": ["authority", "instability", "influence"],
            "concrete_terms": ["coat", "boots", "hands"],
        }
        fields = [
            LexiconJsonFieldInfo(
                name="version",
                type="string",
                description="Data version used by the deterministic abstraction lexicon.",
            ),
            LexiconJsonFieldInfo(
                name="abstract_terms",
                type="string[]",
                description="Lowercase abstract terms used by the abstraction-up indicator.",
            ),
            LexiconJsonFieldInfo(
                name="concrete_terms",
                type="string[]",
                description="Lowercase concrete terms used by the abstraction-up indicator.",
            ),
        ]
        notes = [
            "The micro-indicator pipeline lowercases these terms for deterministic membership checks.",
            "Unknown top-level keys are rejected when saving drafts.",
        ]
    elif kind == "embodiment":
        sample_payload = {
            "version": "0.1",
            "abstract": ["tension", "instability", "conflict"],
            "physical": ["hand", "hands", "shoulder"],
        }
        fields = [
            LexiconJsonFieldInfo(
                name="version",
                type="string",
                description="Data version used by the deterministic embodiment lexicon.",
            ),
            LexiconJsonFieldInfo(
                name="abstract",
                type="string[]",
                description="Lowercase abstract words used on the removed-text side of embodiment shift detection.",
            ),
            LexiconJsonFieldInfo(
                name="physical",
                type="string[]",
                description="Lowercase physical words used on the added-text side of embodiment shift detection.",
            ),
        ]
        notes = [
            "Embodiment shift looks for abstract words removed and physical words added.",
            "Unknown top-level keys are rejected when saving drafts.",
        ]
    else:
        sample_payload = {
            "version": "0.1",
            "scales": {
                "unease_scale": ["uneasy", "tense", "troubled"],
                "confidence_scale": ["hesitant", "steady", "assured"],
            },
        }
        fields = [
            LexiconJsonFieldInfo(
                name="version",
                type="string",
                description="Data version used by the deterministic intensity lexicon.",
            ),
            LexiconJsonFieldInfo(
                name="scales",
                type="object<string,string[]>",
                description="Ordered per-scale word lists used to detect upward and downward intensity moves.",
            ),
        ]
        notes = [
            "Each scale is ordered from lower to higher intensity.",
            "Unknown top-level keys are rejected when saving drafts.",
        ]

    notes.append(
        "Drafts are saved under app/data/drafts and do not affect runtime canonical files."
    )
    return LexiconJsonReference(
        artifact_kind=kind,
        fields=fields,
        sample_json=json.dumps(sample_payload, ensure_ascii=False, indent=2),
        notes=notes,
    )


def _parse_lexicon_payload(raw: str | bytes | dict) -> tuple[LexiconKind, BaseModel]:
    """Validate one deterministic lexicon payload and return its contract kind."""

    if isinstance(raw, (str, bytes)):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc.msg}") from exc
    else:
        parsed = raw

    validators: tuple[tuple[LexiconKind, type[BaseModel]], ...] = (
        ("abstraction", AbstractionLexiconPayload),
        ("embodiment", EmbodimentLexiconPayload),
        ("intensity", IntensityLexiconPayload),
    )
    validation_errors: list[str] = []
    for kind, model_cls in validators:
        try:
            return kind, model_cls.model_validate(parsed)
        except ValidationError as exc:
            validation_errors.append(f"{kind}: {exc.errors()[0]['msg']}")

    raise HTTPException(
        status_code=400,
        detail=(
            "JSON does not match any supported lexicon contract. " + "; ".join(validation_errors)
        ),
    )


def _policy_bundle_reference() -> PolicyBundleReference:
    """Return schema/reference metadata for normalized policy bundle JSON artifacts."""

    sample_bundle = {
        "world_id": "pipeworks_web",
        "version": "0.1.0",
        "source": "mud_server policy package normalized to JSON",
        "policy_hash": None,
        "axes_order": ["physique", "wealth", "health"],
        "axes": {
            "physique": {
                "group": "character",
                "ordering": ["frail", "hunched", "skinny"],
                "thresholds": [
                    {"label": "frail", "min": 0.0, "max": 0.16},
                    {"label": "hunched", "min": 0.17, "max": 0.32},
                    {"label": "skinny", "min": 0.33, "max": 0.48},
                ],
            },
            "wealth": {
                "group": "character",
                "ordering": ["poor", "modest", "well-kept"],
                "thresholds": [
                    {"label": "poor", "min": 0.0, "max": 0.19},
                    {"label": "modest", "min": 0.2, "max": 0.39},
                    {"label": "well-kept", "min": 0.4, "max": 0.59},
                ],
            },
            "health": {
                "group": "character",
                "ordering": ["sickly", "limping", "weary"],
                "thresholds": [
                    {"label": "sickly", "min": 0.0, "max": 0.19},
                    {"label": "limping", "min": 0.2, "max": 0.39},
                    {"label": "weary", "min": 0.4, "max": 0.59},
                ],
            },
        },
        "chat_rules": {
            "channel_multipliers": {"say": 1.0, "yell": 1.5, "whisper": 0.5},
            "min_gap_threshold": 0.05,
            "axes": {
                "physique": {"resolver": "no_effect"},
                "wealth": {"resolver": "no_effect"},
                "health": {"resolver": "shared_drain", "base_magnitude": 0.01},
            },
        },
    }
    return PolicyBundleReference(
        fields=[
            PolicyBundleFieldInfo(
                name="world_id",
                type="string",
                description="World identifier this normalized policy bundle targets.",
            ),
            PolicyBundleFieldInfo(
                name="version",
                type="string",
                description="Policy package version mirrored from the canonical mud-server files.",
            ),
            PolicyBundleFieldInfo(
                name="source",
                type="string",
                description="Provenance note describing how the bundle was derived.",
            ),
            PolicyBundleFieldInfo(
                name="policy_hash",
                type="string|null",
                description="Optional canonical mud-server policy hash, when known.",
            ),
            PolicyBundleFieldInfo(
                name="axes_order",
                type="string[]",
                description="Canonical axis order. Must match the axes object key order exactly.",
            ),
            PolicyBundleFieldInfo(
                name="axes",
                type="object",
                description="Per-axis group, ordinal ordering, and threshold ranges normalized from axes.yaml and thresholds.yaml.",
            ),
            PolicyBundleFieldInfo(
                name="chat_rules",
                type="object",
                description="Chat interaction rules normalized from resolution.yaml.",
            ),
        ],
        sample_json=json.dumps(sample_bundle, ensure_ascii=False, indent=2),
        notes=[
            "This is a local JSON normalization of the mud-server policy package, not a replacement for the canonical YAML files.",
            "Draft saves are validated for axis ordering, threshold ordering, and chat-rule axis coverage before they are written.",
            "Local drafts are stored under app/artifacts/policy_bundles/drafts and never overwrite shipped starter bundles.",
        ],
    )


def _parse_policy_bundle_payload(raw: str | bytes | dict) -> PolicyBundlePayload:
    """Validate one normalized policy bundle JSON payload."""

    if isinstance(raw, (str, bytes)):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc.msg}") from exc
    else:
        parsed = raw

    try:
        return PolicyBundlePayload.model_validate(parsed)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc.errors()[0]["msg"])) from exc


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
            name=row.stem,
            purpose=purpose,
            is_draft=row.is_draft,
            is_active=False,
            origin_path=row.source_path,
        )
        for row in _resolved_prompt_rows(purpose)
    ]
    return LocalPromptArtifactListResponse(
        purpose=purpose,
        prompts=prompts,
        reference=build_prompt_reference(purpose),
    )


def list_local_axis_payload_artifacts() -> LocalAxisPayloadArtifactListResponse:
    """List shipped and draft AxisPayload JSON files under app/examples."""

    payloads: list[AxisPayloadArtifactSummary] = []
    for row in _resolved_axis_payload_rows():
        payload = AxisPayload.model_validate_json(row.path.read_text(encoding="utf-8"))
        payloads.append(
            AxisPayloadArtifactSummary(
                name=row.stem,
                is_draft=row.is_draft,
                origin_path=row.source_path,
                world_id=payload.world_id,
            )
        )

    return LocalAxisPayloadArtifactListResponse(
        payloads=payloads,
        reference=_axis_payload_reference(),
    )


def list_local_lexicon_json_artifacts() -> LocalLexiconJsonArtifactListResponse:
    """List shipped and draft deterministic lexicon JSON files under app/data."""

    lexicons: list[LexiconJsonArtifactSummary] = []
    for row in _resolved_lexicon_rows():
        kind, payload = _parse_lexicon_payload(row.path.read_text(encoding="utf-8"))
        lexicons.append(
            LexiconJsonArtifactSummary(
                name=row.stem,
                artifact_kind=kind,
                is_draft=row.is_draft,
                origin_path=row.source_path,
                version=str(payload.model_dump()["version"]),
            )
        )

    return LocalLexiconJsonArtifactListResponse(
        lexicons=lexicons,
        reference=_lexicon_catalog_reference(),
    )


def list_local_policy_bundle_artifacts() -> LocalPolicyBundleArtifactListResponse:
    """List shipped and draft normalized policy bundle JSON files."""

    bundles: list[PolicyBundleArtifactSummary] = []
    for row in _resolved_policy_bundle_rows():
        payload = _parse_policy_bundle_payload(row.path.read_text(encoding="utf-8"))
        bundles.append(
            PolicyBundleArtifactSummary(
                name=row.stem,
                is_draft=row.is_draft,
                origin_path=row.source_path,
                world_id=payload.world_id,
                version=payload.version,
            )
        )

    return LocalPolicyBundleArtifactListResponse(
        bundles=bundles,
        reference=_policy_bundle_reference(),
    )


def load_local_prompt_artifact(name: str, purpose: PromptPurpose) -> PromptArtifactDocument:
    """Load one local prompt file together with its editor contract."""

    target = next((row for row in _resolved_prompt_rows(purpose) if row.stem == name), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Prompt artifact '{name}' not found.")

    return PromptArtifactDocument(
        name=name,
        purpose=purpose,
        content=target.path.read_text(encoding="utf-8").strip(),
        is_draft=target.is_draft,
        origin_path=target.source_path,
        reference=build_prompt_reference(purpose),
    )


def load_local_axis_payload_artifact(name: str) -> AxisPayloadArtifactDocument:
    """Load one local AxisPayload JSON artifact together with its reference contract."""

    target = next((row for row in _resolved_axis_payload_rows() if row.stem == name), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Axis payload artifact '{name}' not found.")

    payload = AxisPayload.model_validate_json(target.path.read_text(encoding="utf-8"))
    normalized = json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
    return AxisPayloadArtifactDocument(
        name=name,
        content=normalized,
        is_draft=target.is_draft,
        origin_path=target.source_path,
        world_id=payload.world_id,
        reference=_axis_payload_reference(),
    )


def load_local_lexicon_json_artifact(name: str) -> LexiconJsonArtifactDocument:
    """Load one local deterministic lexicon JSON artifact and its reference contract."""

    target = next((row for row in _resolved_lexicon_rows() if row.stem == name), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Lexicon artifact '{name}' not found.")

    kind, payload = _parse_lexicon_payload(target.path.read_text(encoding="utf-8"))
    normalized = json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
    return LexiconJsonArtifactDocument(
        name=name,
        artifact_kind=kind,
        content=normalized,
        is_draft=target.is_draft,
        origin_path=target.source_path,
        version=str(payload.model_dump()["version"]),
        reference=_lexicon_reference(kind),
    )


def load_local_policy_bundle_artifact(name: str) -> PolicyBundleArtifactDocument:
    """Load one normalized policy bundle JSON artifact and its reference contract."""

    target = next((row for row in _resolved_policy_bundle_rows() if row.stem == name), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Policy bundle artifact '{name}' not found.")

    payload = _parse_policy_bundle_payload(target.path.read_text(encoding="utf-8"))
    normalized = json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
    return PolicyBundleArtifactDocument(
        name=name,
        content=normalized,
        is_draft=target.is_draft,
        origin_path=target.source_path,
        world_id=payload.world_id,
        version=payload.version,
        reference=_policy_bundle_reference(),
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


def create_local_axis_payload_draft(
    req: LocalAxisPayloadDraftCreateRequest,
) -> LocalAxisPayloadDraftCreateResponse:
    """Create a new validated AxisPayload JSON draft under app/examples/drafts."""

    draft_name = req.draft_name.strip()
    if not _DRAFT_NAME_RE.fullmatch(draft_name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Draft names must use lowercase letters, numbers, underscores, or hyphens "
                "and must not include a file extension."
            ),
        )

    existing_names = {path.stem for path in _iter_axis_payload_files()}
    if draft_name in existing_names:
        raise HTTPException(
            status_code=409,
            detail=f"An AxisPayload artifact named '{draft_name}' already exists.",
        )

    try:
        parsed = json.loads(req.content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc.msg}") from exc

    payload = AxisPayload.model_validate(parsed)
    drafts_dir = _examples_root() / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    target = drafts_dir / f"{draft_name}.json"
    target.write_text(
        json.dumps(payload.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return LocalAxisPayloadDraftCreateResponse(
        name=draft_name,
        origin_path=_axis_payload_origin_path(target),
        world_id=payload.world_id,
        based_on_name=req.based_on_name,
    )


def create_local_lexicon_json_draft(
    req: LocalLexiconJsonDraftCreateRequest,
) -> LocalLexiconJsonDraftCreateResponse:
    """Create a new validated deterministic lexicon JSON draft under app/data/drafts."""

    draft_name = req.draft_name.strip()
    if not _DRAFT_NAME_RE.fullmatch(draft_name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Draft names must use lowercase letters, numbers, underscores, or hyphens "
                "and must not include a file extension."
            ),
        )

    existing_names = {path.stem for path in _iter_lexicon_json_files()}
    if draft_name in existing_names:
        raise HTTPException(
            status_code=409,
            detail=f"A lexicon artifact named '{draft_name}' already exists.",
        )

    kind, payload = _parse_lexicon_payload(req.content)
    drafts_dir = _data_root() / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    target = drafts_dir / f"{draft_name}.json"
    normalized = json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
    target.write_text(normalized + "\n", encoding="utf-8")

    return LocalLexiconJsonDraftCreateResponse(
        name=draft_name,
        artifact_kind=kind,
        origin_path=_lexicon_origin_path(target),
        version=str(payload.model_dump()["version"]),
        based_on_name=req.based_on_name,
    )


def create_local_policy_bundle_draft(
    req: LocalPolicyBundleDraftCreateRequest,
) -> LocalPolicyBundleDraftCreateResponse:
    """Create a new validated normalized policy bundle JSON draft."""

    draft_name = req.draft_name.strip()
    if not _DRAFT_NAME_RE.fullmatch(draft_name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Draft names must use lowercase letters, numbers, underscores, or hyphens "
                "and must not include a file extension."
            ),
        )

    existing_names = {path.stem for path in _iter_policy_bundle_files()}
    if draft_name in existing_names:
        raise HTTPException(
            status_code=409,
            detail=f"A policy bundle artifact named '{draft_name}' already exists.",
        )

    payload = _parse_policy_bundle_payload(req.content)
    drafts_dir = _policy_bundle_root() / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    target = drafts_dir / f"{draft_name}.json"
    normalized = json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
    target.write_text(normalized + "\n", encoding="utf-8")

    return LocalPolicyBundleDraftCreateResponse(
        name=draft_name,
        origin_path=_policy_bundle_origin_path(target),
        world_id=payload.world_id,
        version=payload.version,
        based_on_name=req.based_on_name,
    )


def create_server_policy_bundle_draft(
    *,
    world_id: str,
    req: ServerPolicyBundleDraftCreateRequest,
    mud_client: MudServerClient,
) -> ServerPolicyBundleDraftCreateResponse:
    """Validate and forward a create-only mud-server policy bundle draft request."""

    payload = _parse_policy_bundle_payload(req.content)
    if payload.world_id != world_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Policy bundle world_id must match the selected mud-server world before "
                "creating a server draft."
            ),
        )

    data = mud_client.create_world_policy_bundle_draft(
        world_id=world_id,
        draft_name=req.draft_name.strip(),
        content=payload.model_dump(),
        based_on_name=req.based_on_name,
    )
    return ServerPolicyBundleDraftCreateResponse.model_validate(data)


def list_server_policy_bundle_artifacts(
    world_id: str,
    mud_client: MudServerClient,
) -> ServerPolicyBundleArtifactListResponse:
    """Return mud-server policy bundle drafts for one selected world."""

    data = mud_client.world_policy_bundle_drafts(world_id)
    bundles = [
        PolicyBundleArtifactSummary(
            name=str(entry.get("name") or ""),
            is_draft=True,
            origin_path=str(entry.get("origin_path") or ""),
            world_id=str(entry.get("world_id") or world_id),
            version=str(entry.get("version") or ""),
        )
        for entry in data.get("drafts", [])
    ]
    return ServerPolicyBundleArtifactListResponse(
        world_id=world_id,
        bundles=bundles,
        reference=_policy_bundle_reference(),
    )


def load_server_policy_bundle_draft_artifact(
    *,
    world_id: str,
    draft_name: str,
    mud_client: MudServerClient,
) -> PolicyBundleArtifactDocument:
    """Load one mud-server policy bundle draft into the editor document shape."""

    data = mud_client.world_policy_bundle_draft(world_id, draft_name)
    payload = _parse_policy_bundle_payload(data.get("content") or {})
    normalized = json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
    return PolicyBundleArtifactDocument(
        name=str(data.get("name") or draft_name),
        content=normalized,
        is_draft=True,
        origin_path=str(data.get("origin_path") or ""),
        world_id=payload.world_id,
        version=payload.version,
        reference=_policy_bundle_reference(),
    )


def promote_server_policy_bundle_draft(
    *,
    world_id: str,
    draft_name: str,
    req: ServerPolicyBundleDraftPromoteRequest,
    mud_client: MudServerClient,
) -> ServerPolicyBundleDraftPromoteResponse:
    """Forward an explicit mud-server policy-bundle promotion request."""

    del req
    data = mud_client.promote_world_policy_bundle_draft(
        world_id=world_id,
        draft_name=draft_name,
    )
    return ServerPolicyBundleDraftPromoteResponse.model_validate(data)


def get_server_policy_bundle_artifact(
    world_id: str,
    mud_client: MudServerClient,
) -> PolicyBundleArtifactDocument:
    """Build a normalized policy bundle document from the mud server's lab endpoint."""

    data = mud_client.world_policy_bundle(world_id)
    payload = _parse_policy_bundle_payload(
        {
            "world_id": data.get("world_id"),
            "version": data.get("version"),
            "source": data.get("source"),
            "policy_hash": data.get("policy_hash"),
            "axes_order": data.get("axes_order"),
            "axes": data.get("axes"),
            "chat_rules": data.get("chat_rules"),
        }
    )
    normalized = json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
    return PolicyBundleArtifactDocument(
        name=f"{world_id}_policy_bundle",
        content=normalized,
        is_draft=False,
        origin_path=", ".join(data.get("source_files") or []),
        world_id=payload.world_id,
        version=payload.version,
        reference=_policy_bundle_reference(),
    )


def create_server_prompt_draft(
    *,
    world_id: str,
    req: ServerPromptDraftCreateRequest,
    mud_client: MudServerClient,
) -> ServerPromptDraftCreateResponse:
    """Validate and forward a create-only mud-server prompt draft request."""

    data = mud_client.create_world_prompt_draft(
        world_id=world_id,
        draft_name=req.draft_name.strip(),
        content=req.content.rstrip() + "\n",
        based_on_name=req.based_on_name,
    )
    return ServerPromptDraftCreateResponse.model_validate(data)


def promote_server_prompt_draft(
    *,
    world_id: str,
    draft_name: str,
    req: ServerPromptDraftPromoteRequest,
    mud_client: MudServerClient,
) -> ServerPromptDraftPromoteResponse:
    """Forward an explicit mud-server prompt promotion request."""

    data = mud_client.promote_world_prompt_draft(
        world_id=world_id,
        draft_name=draft_name,
        target_name=req.target_name.strip(),
    )
    return ServerPromptDraftPromoteResponse.model_validate(data)


def list_server_prompt_artifacts(
    world_id: str,
    mud_client: MudServerClient,
) -> ServerPromptArtifactListResponse:
    """Return mud-server prompt drafts for one selected world."""

    world_cfg = mud_client.world_config(world_id)
    active_axes = list(world_cfg.get("active_axes") or [])
    data = mud_client.world_prompt_drafts(world_id)
    prompts = [
        PromptArtifactSummary(
            name=str(entry.get("name") or ""),
            purpose="chat_translation",
            is_draft=True,
            is_active=False,
            origin_path=str(entry.get("origin_path") or ""),
        )
        for entry in data.get("drafts", [])
    ]
    return ServerPromptArtifactListResponse(
        world_id=world_id,
        prompts=prompts,
        reference=build_prompt_reference(
            "chat_translation",
            source_mode="server",
            world_id=world_id,
            active_axes=active_axes,
        ),
    )


def load_server_prompt_draft_artifact(
    *,
    world_id: str,
    draft_name: str,
    mud_client: MudServerClient,
) -> PromptArtifactDocument:
    """Load one mud-server prompt draft into the editor document shape."""

    world_cfg = mud_client.world_config(world_id)
    active_axes = list(world_cfg.get("active_axes") or [])
    data = mud_client.world_prompt_draft(world_id, draft_name)
    return PromptArtifactDocument(
        name=str(data.get("name") or draft_name),
        purpose="chat_translation",
        content=str(data.get("content") or ""),
        is_draft=True,
        origin_path=str(data.get("origin_path") or ""),
        reference=build_prompt_reference(
            "chat_translation",
            source_mode="server",
            world_id=world_id,
            active_axes=active_axes,
        ),
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
