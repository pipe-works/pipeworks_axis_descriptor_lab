"""
app.schema
-----------------------------------------------------------------------------
Domain-organised Pydantic v2 models for the Axis Descriptor Lab API.

This package replaces the previous monolithic ``app/schema.py`` module while
preserving the original import surface.  Existing imports such as
``from app.schema import AxisPayload`` remain valid because this package
re-exports every public schema name from the domain submodules below.

Submodules
----------
- ``axis``     – shared axis primitives used across multiple endpoints.
- ``generate`` – generate/log request and response models.
- ``save``     – save/import and manifest models.
- ``analysis`` – signal-isolation and transformation-map models.
- ``artifact`` – Artifact Editor prompt-manifest and draft models.
- ``chat``     – chat translation, chat save, and chat import models.
- ``mud``      – mud-server proxy request and response models.
"""

from app.schema.artifact import (
    AxisPayloadArtifactDocument,
    AxisPayloadArtifactSummary,
    AxisPayloadFieldInfo,
    AxisPayloadReference,
    ArtifactPlaceholder,
    ArtifactPromptReference,
    LexiconJsonArtifactDocument,
    LexiconJsonArtifactSummary,
    LexiconJsonFieldInfo,
    LexiconJsonReference,
    LocalAxisPayloadArtifactListResponse,
    LocalAxisPayloadDraftCreateRequest,
    LocalAxisPayloadDraftCreateResponse,
    LocalLexiconJsonArtifactListResponse,
    LocalLexiconJsonDraftCreateRequest,
    LocalLexiconJsonDraftCreateResponse,
    LocalPolicyBundleArtifactListResponse,
    LocalPolicyBundleDraftCreateRequest,
    LocalPolicyBundleDraftCreateResponse,
    PolicyBundleArtifactDocument,
    PolicyBundleArtifactSummary,
    PolicyBundleFieldInfo,
    PolicyBundleReference,
    LocalPromptArtifactListResponse,
    LocalPromptDraftCreateRequest,
    LocalPromptDraftCreateResponse,
    PromptArtifactDocument,
    PromptArtifactSummary,
    ServerPromptArtifactListResponse,
    ServerPromptDraftCreateRequest,
    ServerPromptDraftCreateResponse,
    ServerPromptDraftPromoteRequest,
    ServerPromptDraftPromoteResponse,
    ServerPolicyBundleDraftCreateRequest,
    ServerPolicyBundleDraftCreateResponse,
    ServerPolicyBundleArtifactListResponse,
    ServerPolicyBundleDraftPromoteRequest,
    ServerPolicyBundleDraftPromoteResponse,
    ServerPromptManifestResponse,
)
from app.schema.analysis import (
    DeltaRequest,
    DeltaResponse,
    IndicatorConfig,
    TransformationMapRequest,
    TransformationMapResponse,
    TransformationMapRow,
)
from app.schema.axis import AxisPayload, AxisValue
from app.schema.chat import (
    ChatCharacterInput,
    ChatImportResponse,
    ChatLogEntry,
    ChatSaveRequest,
    ChatSaveResponse,
    ChatTranslationRequest,
    ChatTranslationResponse,
    ChatTranslationResult,
)
from app.schema.generate import GenerateRequest, GenerateResponse, LogEntry
from app.schema.mud import (
    MudLoginRequest,
    MudLoginResponse,
    MudModeOption,
    MudModeRequest,
    MudModeResponse,
    MudSelectWorldRequest,
    MudSessionResponse,
)
from app.schema.save import ImportResponse, ManifestFileEntry, SaveRequest, SaveResponse

__all__ = [
    "ArtifactPlaceholder",
    "ArtifactPromptReference",
    "AxisPayloadArtifactDocument",
    "AxisPayloadArtifactSummary",
    "AxisPayloadFieldInfo",
    "AxisPayloadReference",
    "AxisPayload",
    "AxisValue",
    "ChatCharacterInput",
    "ChatImportResponse",
    "ChatLogEntry",
    "ChatSaveRequest",
    "ChatSaveResponse",
    "ChatTranslationRequest",
    "ChatTranslationResponse",
    "ChatTranslationResult",
    "DeltaRequest",
    "DeltaResponse",
    "GenerateRequest",
    "GenerateResponse",
    "ImportResponse",
    "IndicatorConfig",
    "LexiconJsonArtifactDocument",
    "LexiconJsonArtifactSummary",
    "LexiconJsonFieldInfo",
    "LexiconJsonReference",
    "LogEntry",
    "LocalPromptArtifactListResponse",
    "LocalPromptDraftCreateRequest",
    "LocalPromptDraftCreateResponse",
    "LocalAxisPayloadArtifactListResponse",
    "LocalAxisPayloadDraftCreateRequest",
    "LocalAxisPayloadDraftCreateResponse",
    "LocalLexiconJsonArtifactListResponse",
    "LocalLexiconJsonDraftCreateRequest",
    "LocalLexiconJsonDraftCreateResponse",
    "LocalPolicyBundleArtifactListResponse",
    "LocalPolicyBundleDraftCreateRequest",
    "LocalPolicyBundleDraftCreateResponse",
    "ManifestFileEntry",
    "MudLoginRequest",
    "MudLoginResponse",
    "MudModeOption",
    "MudModeRequest",
    "MudModeResponse",
    "MudSelectWorldRequest",
    "MudSessionResponse",
    "PolicyBundleArtifactDocument",
    "PolicyBundleArtifactSummary",
    "PolicyBundleFieldInfo",
    "PolicyBundleReference",
    "PromptArtifactDocument",
    "PromptArtifactSummary",
    "ServerPromptArtifactListResponse",
    "ServerPromptDraftCreateRequest",
    "ServerPromptDraftCreateResponse",
    "ServerPromptDraftPromoteRequest",
    "ServerPromptDraftPromoteResponse",
    "SaveRequest",
    "SaveResponse",
    "ServerPolicyBundleDraftCreateRequest",
    "ServerPolicyBundleDraftCreateResponse",
    "ServerPolicyBundleDraftPromoteRequest",
    "ServerPolicyBundleDraftPromoteResponse",
    "ServerPolicyBundleArtifactListResponse",
    "ServerPromptManifestResponse",
    "TransformationMapRequest",
    "TransformationMapResponse",
    "TransformationMapRow",
]
