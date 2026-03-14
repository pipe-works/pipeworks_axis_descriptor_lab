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
- ``chat``     – chat translation, chat save, and chat import models.
- ``mud``      – mud-server proxy request and response models.
"""

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
    MudPipelineBootstrapResponse,
    MudCompileImagePromptRequest,
    MudImagePolicyBundleResponse,
    MudLoginRequest,
    MudLoginResponse,
    MudModeOption,
    MudModeRequest,
    MudModeResponse,
    MudPipelinePolicySource,
    MudPipelinePolicySourceReference,
    MudPipelineGenerateConditionAxisEntityInputs,
    MudPipelineGenerateConditionAxisIdentityInputs,
    MudPipelineGenerateConditionAxisInputs,
    MudPipelineGenerateConditionAxisRequest,
    MudPipelineResolveRequest,
    MudPipelineResolveResponse,
    MudPipelineRuntimeOptions,
    MudPipelineSelectedBlocks,
    MudSelectWorldRequest,
    MudSessionResponse,
)
from app.schema.save import ImportResponse, ManifestFileEntry, SaveRequest, SaveResponse

__all__ = [
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
    "LogEntry",
    "ManifestFileEntry",
    "MudCompileImagePromptRequest",
    "MudImagePolicyBundleResponse",
    "MudLoginRequest",
    "MudLoginResponse",
    "MudModeOption",
    "MudModeRequest",
    "MudModeResponse",
    "MudPipelineBootstrapResponse",
    "MudPipelineGenerateConditionAxisEntityInputs",
    "MudPipelineGenerateConditionAxisIdentityInputs",
    "MudPipelineGenerateConditionAxisInputs",
    "MudPipelineGenerateConditionAxisRequest",
    "MudPipelinePolicySource",
    "MudPipelinePolicySourceReference",
    "MudPipelineResolveRequest",
    "MudPipelineResolveResponse",
    "MudPipelineRuntimeOptions",
    "MudPipelineSelectedBlocks",
    "MudSelectWorldRequest",
    "MudSessionResponse",
    "SaveRequest",
    "SaveResponse",
    "TransformationMapRequest",
    "TransformationMapResponse",
    "TransformationMapRow",
]
