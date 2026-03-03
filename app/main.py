"""
app/main.py
-----------------------------------------------------------------------------
FastAPI application entrypoint for the Axis Descriptor Lab.

This module is a **thin routing layer** — each route handler orchestrates
calls to domain modules and returns the result.  All business logic lives
in dedicated modules:

Domain modules
~~~~~~~~~~~~~~
- ``pipeworks_ipc``        – IPC normalisation and SHA-256 hash functions (shared library).
- ``app.schema``           – Pydantic v2 request / response models.
- ``app.chat_renderer``    – Synchronous HTTP wrapper around Ollama.
- ``app.signal_isolation`` – NLP pipeline for content-word delta.
- ``app.transformation_map`` – Clause-level sentence alignment and diffing.
- ``app.micro_indicators`` – Structural pattern indicators for transformation map rows.
- ``app.save_package``     – Manifest builder, zip archive, import/export.
- ``app.relabel_policy``   – Policy table and score-to-label mapping.
- ``app.save_formatting``  – Markdown builders and folder-name generator.
- ``app.file_loaders``     – Example and prompt file loading/listing.
- ``app.mud_server_client`` – Synchronous HTTP client for the mud server lab API.

Run with:
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8242

Endpoints
---------
GET  /                         → serves index.html
GET  /api/examples             → list of available example names
GET  /api/examples/{name}      → returns a single example JSON payload
GET  /api/prompts              → list of available prompt names (optionally by purpose)
GET  /api/prompts/{name}       → returns a single prompt's text content
GET  /api/models               → returns locally available Ollama models
POST /api/generate             → send axis payload to Ollama, return description
POST /api/log                  → persist a run log entry to logs/run_log.jsonl
POST /api/relabel              → (optional) recompute labels from policy rules
POST /api/analyze-delta        → content-word delta between two texts
POST /api/transformation-map   → clause-level replacement pairs
GET  /api/system-prompt        → return the default system prompt as plain text
POST /api/save                 → save session state to a timestamped data/ subfolder
GET  /api/save/{name}/export   → download a save package as a zip
POST /api/import               → import a save package from a zip upload
POST /api/import_chat          → import a chat save package from a zip upload
POST /api/mud/login            → proxy login to mud server
POST /api/mud/logout           → clear mud server session
GET  /api/mud/mode             → return runtime chat mode + available options
POST /api/mud/mode             → switch runtime chat mode
GET  /api/mud/session          → return auth status + translation mode
GET  /api/mud/worlds           → proxy list worlds from mud server
GET  /api/mud/world-config/{id}→ proxy world config from mud server
GET  /api/artifacts/local/chat-prompts          → list local prompt artifacts
GET  /api/artifacts/local/chat-prompts/{name}   → load one local prompt artifact
POST /api/artifacts/local/chat-prompts/drafts   → create a new local prompt draft
GET  /api/artifacts/local/axis-payloads         → list local AxisPayload JSON artifacts
GET  /api/artifacts/local/axis-payloads/{name}  → load one local AxisPayload JSON artifact
POST /api/artifacts/local/axis-payloads/drafts  → create a new local AxisPayload JSON draft
GET  /api/artifacts/local/policy-bundles        → list local normalized policy bundle artifacts
GET  /api/artifacts/local/policy-bundles/{name} → load one local normalized policy bundle artifact
POST /api/artifacts/local/policy-bundles/drafts → create a new local normalized policy bundle draft
GET  /api/artifacts/server/chat-prompts/{id}    → server-backed canonical prompt manifest
GET  /api/artifacts/server/policy-bundles/{id}  → server-backed canonical policy bundle artifact
POST /api/mud/select-world     → store selected world_id

Architecture notes
------------------
- All blocking I/O (file reads, Ollama HTTP calls) lives in regular ``def``
  route handlers.  FastAPI automatically runs those in a threadpool so the
  async event loop is never blocked.
- Static files are served by Starlette's StaticFiles middleware.
- Jinja2Templates renders index.html (single page — the JS takes over).
- A simple JSONL log file under ``logs/`` provides the Pipe-Works audit
  trail without any database dependency.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from pipeworks_ipc import (
    compute_ipc_id,
    compute_output_hash,
    compute_system_prompt_hash,
    payload_hash,
)
from app.file_loaders import (
    list_example_names,
    list_prompt_names,
    load_default_prompt,
    load_example,
    load_prompt,
)
from app.chat_renderer import OLLAMA_HOST, ChatRenderer, close_all_clients
from app.config import APP_VERSION as _APP_VERSION
from app.config import DATA_DIR as _DATA_DIR
from app.config import DEFAULT_MODEL as _DEFAULT_MODEL
from app.config import HERE as _HERE
from app.config import LOGS_DIR as _LOGS_DIR
from app.relabel_policy import apply_relabel_policy
from app.save_package import (
    extract_body_text,
    extract_fenced_code,
    parse_game_log_md,
    validate_and_extract_zip,
    MAX_UPLOAD_SIZE,
)
from app.mud_server_client import close_all_mud_clients, compute_translation_mode
from app.schema import (
    AxisPayload,
    ChatImportResponse,
    DeltaRequest,
    DeltaResponse,
    GenerateRequest,
    GenerateResponse,
    ImportResponse,
    LogEntry,
    TransformationMapRequest,
    TransformationMapResponse,
    TransformationMapRow,
)
from app.micro_indicators import IndicatorConfig as _IndicatorConfig
from app.routes_chat import router as chat_router
from app.routes_artifact_editor import router as artifact_editor_router
from app.routes_mud import router as mud_router
from app.routes_save import router as save_router
from app.micro_indicators import classify_rows
from app.signal_isolation import compute_delta
from app.transformation_map import compute_transformation_map

# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"

# Create runtime output directories if they don't exist (production
# deployments may not have committed .gitkeep files).
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE = _LOGS_DIR / "run_log.jsonl"

# -----------------------------------------------------------------------------
# FastAPI app + middleware
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Axis Descriptor Lab",
    description=(
        "Tiny web tool for testing how small LLMs (via Ollama) produce "
        "non-authoritative descriptive text from a deterministic axis payload."
    ),
    version=_APP_VERSION,
)


def close_runtime_clients() -> None:
    """Close all shared HTTP clients created by the application runtime."""
    close_all_clients()
    close_all_mud_clients()


# Close shared HTTP client pools on shutdown to release connections cleanly.
app.add_event_handler("shutdown", close_runtime_clients)

app.include_router(chat_router)
app.include_router(artifact_editor_router)
app.include_router(mud_router, prefix="/api/mud")
app.include_router(save_router)

# Serve everything under /static/ directly from the filesystem.
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Jinja2 for the single HTML page.
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request) -> HTMLResponse:
    """
    Serve the single-page application shell.

    Passes the default model name and the list of locally available Ollama
    models into the Jinja2 template so the frontend can pre-populate its
    model selector without an extra API round-trip.
    """
    available_models = ChatRenderer.list_models()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_model": _DEFAULT_MODEL,
            "available_models": available_models,
            "ollama_host": OLLAMA_HOST,
            "app_version": _APP_VERSION,
            "translation_mode": compute_translation_mode(),
        },
    )


@app.get("/api/examples", summary="List available example names")
def list_examples() -> list[str]:
    """
    Return a sorted list of example names (without the .json extension) that
    are stored in app/examples/.

    The frontend uses this to populate its example dropdown.
    """
    return list_example_names()


@app.get("/api/examples/{name}", summary="Get a named example payload")
def get_example(name: str) -> dict:
    """
    Return the parsed JSON for a named example.

    Parameters
    ----------
    name : Example stem, e.g. "proud_operator".

    Returns
    -------
    The raw example JSON object (validated loosely by Pydantic when the
    frontend loads it into the textarea).
    """
    return load_example(name)


@app.get("/api/prompts", summary="List available prompt names")
def list_prompts(
    purpose: Literal["character_description", "chat_translation"] | None = None,
) -> list[str]:
    """
    Return a sorted list of prompt names (without the .txt extension) that
    are stored in the grouped ``app/prompts/`` tree.

    The frontend uses this to populate its prompt library dropdown, allowing
    users to browse only the prompt family relevant to the current page.

    Parameters
    ----------
    purpose : {"character_description", "chat_translation"} | None
        Optional prompt-group filter. When omitted, prompt names from every
        local prompt group are returned.
    """
    return list_prompt_names(purpose)


@app.get(
    "/api/prompts/{name}",
    response_class=PlainTextResponse,
    summary="Get a named prompt text",
)
def get_prompt(
    name: str,
    purpose: Literal["character_description", "chat_translation"] | None = None,
) -> str:
    """
    Return the text content of a named prompt file as plain text.

    Uses ``PlainTextResponse`` to match the existing ``/api/system-prompt``
    endpoint pattern.  The frontend loads this into the system prompt
    override textarea.

    Parameters
    ----------
    name : Prompt stem, e.g. "system_prompt_v01".
    purpose : {"character_description", "chat_translation"} | None
        Optional prompt-group filter. When provided, lookup is limited to
        that prompt family.

    Returns
    -------
    The raw prompt text (plain text, not JSON).
    """
    return load_prompt(name, purpose)


@app.get("/api/models", summary="List locally available Ollama models")
def get_models(host: str | None = None) -> list[str]:
    """
    Query an Ollama instance and return all model names it has pulled.

    Parameters
    ----------
    host : Optional Ollama server URL (query param).  When omitted the
           server-side default (``OLLAMA_HOST`` env var) is used.

    Returns an empty list if Ollama is unreachable, allowing the frontend to
    fall back to a manual text input.
    """
    return ChatRenderer.list_models(host=host)


@app.post(
    "/api/generate",
    response_model=GenerateResponse,
    summary="Generate a descriptive paragraph from an axis payload",
)
def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Core endpoint: serialise the axis payload to JSON, attach the system
    prompt, call Ollama, and return the generated paragraph.

    The system prompt is taken from the request body if provided; otherwise
    the default Character Description prompt is loaded from
    ``app/prompts/character_description/``. This lets the frontend
    experiment with custom prompts without restarting the server.

    Parameters
    ----------
    req : GenerateRequest – full request body (payload + model settings).

    Returns
    -------
    GenerateResponse containing the generated text and metadata.

    Raises
    ------
    HTTPException(502) if Ollama returns an HTTP error.
    HTTPException(504) if the Ollama request times out.
    HTTPException(500) for any other unexpected error.
    """
    system_prompt = req.system_prompt or load_default_prompt()

    # Serialise the payload as pretty-printed JSON – this is the "user turn"
    # the LLM receives.  The prompt instructs it not to reference JSON by name.
    user_json_str = json.dumps(req.payload.model_dump(), ensure_ascii=False, indent=2)

    try:
        # Forward the payload's seed to Ollama's options.seed so the model
        # pins its RNG during token sampling.  This makes generation
        # deterministic: identical IPC inputs → identical output text.
        # The optional ollama_host allows the frontend to target a different
        # Ollama instance without changing the server's environment variable.
        text, usage = ChatRenderer(
            host=req.ollama_host or OLLAMA_HOST,
            model=req.model,
            temperature=req.temperature,
            seed=req.payload.seed,
            max_tokens=req.max_tokens,
        ).generate(system_prompt, user_json_str)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out.  Is the model loaded?",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error calling Ollama: {exc}"
        ) from exc

    # -- Compute Interpretive Provenance Chain (IPC) hashes --------------
    # These four hashes fingerprint the complete generation context so that
    # identical runs can be detected and prompt drift can be audited.
    input_hash = payload_hash(req.payload)
    sp_hash = compute_system_prompt_hash(system_prompt)
    out_hash = compute_output_hash(text)
    ipc = compute_ipc_id(
        input_hash=input_hash,
        system_prompt_hash=sp_hash,
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        seed=req.payload.seed,
    )

    return GenerateResponse(
        text=text,
        model=req.model,
        temperature=req.temperature,
        usage=usage,
        input_hash=input_hash,
        system_prompt_hash=sp_hash,
        output_hash=out_hash,
        ipc_id=ipc,
    )


@app.post("/api/log", response_model=LogEntry, summary="Persist a run log entry")
def log_run(
    payload: AxisPayload,
    output: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str | None = None,
) -> LogEntry:
    """
    Append a structured log entry to logs/run_log.jsonl.

    Each line in the JSONL file is one complete LogEntry serialised as compact
    JSON.  The file can be opened in any JSONL-aware tool (jq, pandas, etc.)
    for drift analysis.

    When ``system_prompt`` is provided, the entry includes IPC hashes
    (``system_prompt_hash``, ``output_hash``, ``ipc_id``).  When omitted,
    ``output_hash`` is still computed but the prompt-dependent fields are
    set to None for backward compatibility with older frontend versions.

    Parameters
    ----------
    payload       : The AxisPayload used in the run.
    output        : The LLM-generated text.
    model         : Ollama model identifier.
    temperature   : Sampling temperature used.
    max_tokens    : Token budget used.
    system_prompt : The system prompt used (optional).  When provided,
                    enables full IPC chain in the log entry.

    Returns
    -------
    The complete LogEntry that was persisted.
    """
    input_hash = payload_hash(payload)

    # Always compute the output hash — the output text is always available.
    out_hash = compute_output_hash(output)

    # Compute prompt-dependent hashes only when the system prompt is provided.
    # This preserves backward compatibility: older frontends that don't send
    # the prompt will still produce valid log entries with null IPC fields.
    sp_hash: str | None = None
    ipc: str | None = None
    if system_prompt is not None:
        sp_hash = compute_system_prompt_hash(system_prompt)
        ipc = compute_ipc_id(
            input_hash=input_hash,
            system_prompt_hash=sp_hash,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=payload.seed,
        )

    entry = LogEntry(
        input_hash=input_hash,
        payload=payload,
        output=output,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timestamp=datetime.now(timezone.utc).isoformat(),
        system_prompt_hash=sp_hash,
        output_hash=out_hash,
        ipc_id=ipc,
    )

    # Append as a single compact JSON line.
    with _LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json() + "\n")

    return entry


@app.post(
    "/api/relabel",
    response_model=AxisPayload,
    summary="Recompute axis labels from policy score mappings",
)
def relabel(payload: AxisPayload) -> AxisPayload:
    """
    Optional "auto-label" endpoint (Strategy 2 from spec §9.1).

    Applies a simple piecewise score → label mapping for each known axis so
    the lab can demonstrate *policy drift* as well as LLM drift.  Unknown
    axes are left unchanged.

    Delegates to :func:`app.relabel_policy.apply_relabel_policy` which owns
    the policy table and mapping logic.

    Parameters
    ----------
    payload : Current AxisPayload (axes with scores, possibly stale labels).

    Returns
    -------
    Updated AxisPayload with labels recomputed from scores.
    """
    return apply_relabel_policy(payload)


# -----------------------------------------------------------------------------
# POST /api/analyze-delta
# -----------------------------------------------------------------------------


@app.post(
    "/api/analyze-delta",
    response_model=DeltaResponse,
    summary="Compute content-word delta between baseline and current text",
)
def analyze_delta(req: DeltaRequest) -> DeltaResponse:
    """
    Signal Isolation Layer endpoint.

    Takes two text strings (baseline A and current B), runs both through
    the NLP pipeline (tokenise, lemmatise, filter stopwords), and returns
    the set difference as alphabetically sorted word lists.

    This endpoint surfaces meaningful lexical pivots by filtering
    structural noise.  It is deterministic: same inputs always produce
    the same outputs.

    The LLM is not involved — this is pure programmatic text analysis.

    Parameters
    ----------
    req : DeltaRequest
        Contains ``baseline_text`` and ``current_text``.

    Returns
    -------
    DeltaResponse
        Alphabetically sorted ``removed`` and ``added`` content-lemma
        lists.
    """
    removed, added = compute_delta(req.baseline_text, req.current_text)
    return DeltaResponse(removed=removed, added=added)


# -----------------------------------------------------------------------------
# POST /api/transformation-map
# -----------------------------------------------------------------------------


@app.post(
    "/api/transformation-map",
    response_model=TransformationMapResponse,
    summary="Compute clause-level replacement pairs with micro-indicators",
)
def transformation_map(req: TransformationMapRequest) -> TransformationMapResponse:
    """
    Transformation Map endpoint with micro-indicators.

    Takes two text strings (baseline A and current B), runs sentence-aware
    alignment followed by token-level diffing, and returns clause-level
    replacement pairs.  Each row is then classified with structural
    micro-indicators (compression, embodiment shift, intensity change, etc.)
    using deterministic heuristics.

    This fills the gap between word-level diff (too granular) and
    content-word delta (structure-blind) by showing *what chunk of text
    was replaced by what chunk* at the clause scale, annotated with
    structural pattern labels.

    The LLM is not involved — this is pure programmatic text analysis.

    Parameters
    ----------
    req : TransformationMapRequest
        Contains ``baseline_text``, ``current_text``, and optional
        ``indicator_config`` for tuning heuristic thresholds.

    Returns
    -------
    TransformationMapResponse
        Ordered list of ``TransformationMapRow`` replacement pairs,
        each annotated with zero or more micro-indicator labels.
    """
    rows = compute_transformation_map(
        req.baseline_text, req.current_text, include_all=req.include_all
    )

    # Convert the Pydantic IndicatorConfig to the module's dataclass.
    config = None
    if req.indicator_config is not None:
        config = _IndicatorConfig(
            compression_ratio=req.indicator_config.compression_ratio,
            expansion_ratio=req.indicator_config.expansion_ratio,
            min_tokens=req.indicator_config.min_tokens,
            modality_density_threshold=req.indicator_config.modality_density_threshold,
            enabled=(
                tuple(req.indicator_config.enabled)
                if req.indicator_config.enabled is not None
                else None
            ),
        )

    indicators = classify_rows(rows, config=config)

    return TransformationMapResponse(
        rows=[
            TransformationMapRow(
                removed=row["removed"],
                added=row["added"],
                indicators=ind,
            )
            for row, ind in zip(rows, indicators)
        ]
    )


# -----------------------------------------------------------------------------
# POST /api/import
# -----------------------------------------------------------------------------


@app.post(
    "/api/import",
    response_model=ImportResponse,
    summary="Import a save package from a zip upload",
)
async def import_save(file: UploadFile) -> ImportResponse:
    """
    Accept an uploaded save-package zip, validate it, and return structured
    state for the frontend to restore a complete session.

    The endpoint reads the uploaded file, enforces a maximum upload size,
    validates the zip structure and manifest checksums (if present), then
    extracts plain text from the Markdown files so the frontend can
    populate the UI directly without parsing.

    Parameters
    ----------
    file : The uploaded zip file (multipart form data).

    Returns
    -------
    ImportResponse : Everything the frontend needs to restore session state.

    Raises
    ------
    HTTPException(400) : If the file is not a valid zip, exceeds size limits,
                         or fails checksum validation.
    HTTPException(422) : If required files (metadata.json, payload.json,
                         system_prompt.md) are missing from the zip.
    """
    # Read the uploaded file and enforce the size limit.
    zip_bytes = await file.read()
    if len(zip_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Upload size ({len(zip_bytes):,} bytes) exceeds the "
                f"{MAX_UPLOAD_SIZE:,}-byte limit."
            ),
        )

    # Validate and extract the zip contents.
    try:
        extracted, warnings = validate_and_extract_zip(zip_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- Parse required files -------------------------------------------- #

    # metadata.json — provides model, temperature, max_tokens, seed, etc.
    if "metadata.json" not in extracted:
        raise HTTPException(
            status_code=422,
            detail="Missing required file: metadata.json",
        )
    try:
        metadata = json.loads(extracted["metadata.json"].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"metadata.json is not valid JSON: {exc}",
        ) from exc

    # payload.json — the authoritative axis data.
    if "payload.json" not in extracted:
        raise HTTPException(
            status_code=422,
            detail="Missing required file: payload.json",
        )
    try:
        payload_dict = json.loads(extracted["payload.json"].decode("utf-8"))
        payload = AxisPayload(**payload_dict)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"payload.json is invalid: {exc}",
        ) from exc

    # system_prompt.md — the prompt text (required).
    if "system_prompt.md" not in extracted:
        raise HTTPException(
            status_code=422,
            detail="Missing required file: system_prompt.md",
        )
    system_prompt = extract_fenced_code(extracted["system_prompt.md"].decode("utf-8"))

    # -- Parse optional files -------------------------------------------- #

    output: str | None = None
    if "output.md" in extracted:
        output = extract_body_text(extracted["output.md"].decode("utf-8"))

    baseline: str | None = None
    if "baseline.md" in extracted:
        baseline = extract_body_text(extracted["baseline.md"].decode("utf-8"))

    # -- Determine manifest validity ------------------------------------- #
    # If validation passed without raising (no checksum mismatch),
    # the manifest is valid.  This is always True here because
    # validate_and_extract_zip raises on mismatch.
    manifest_valid = True

    return ImportResponse(
        folder_name=metadata.get("folder_name", "unknown"),
        metadata=metadata,
        payload=payload,
        system_prompt=system_prompt,
        output=output,
        baseline=baseline,
        model=metadata.get("model", "unknown"),
        temperature=metadata.get("temperature", 0.2),
        max_tokens=metadata.get("max_tokens", 120),
        manifest_valid=manifest_valid,
        files=sorted(extracted.keys()),
        warnings=warnings,
    )


# -----------------------------------------------------------------------------
# POST /api/import_chat
# -----------------------------------------------------------------------------


@app.post(
    "/api/import_chat",
    response_model=ChatImportResponse,
    summary="Import a chat save package from a zip upload",
)
async def import_chat_save(file: UploadFile) -> ChatImportResponse:
    """
    Accept an uploaded chat save-package zip and return structured state
    for the frontend to restore a complete chat translation session.

    Extracts character payloads, model settings, system prompt, and the
    historical game log so the frontend can rebuild sliders, settings, and
    the in-game log panel without any further parsing.

    Parameters
    ----------
    file : The uploaded zip file (multipart form data).

    Returns
    -------
    ChatImportResponse : Everything the frontend needs to restore chat state.

    Raises
    ------
    HTTPException(400) : If the file is not a valid zip, exceeds size limits,
                         or fails checksum validation.
    HTTPException(422) : If metadata.json is missing from the zip.
    """
    zip_bytes = await file.read()
    if len(zip_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Upload size ({len(zip_bytes):,} bytes) exceeds the "
                f"{MAX_UPLOAD_SIZE:,}-byte limit."
            ),
        )

    try:
        extracted, warnings = validate_and_extract_zip(zip_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- Parse required file ------------------------------------------------- #

    if "metadata.json" not in extracted:
        raise HTTPException(status_code=422, detail="Missing required file: metadata.json")
    try:
        metadata = json.loads(extracted["metadata.json"].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"metadata.json is not valid JSON: {exc}"
        ) from exc

    # -- Parse system_prompt.md (required; server default used as fallback) -- #

    system_prompt = ""
    if "system_prompt.md" in extracted:
        system_prompt = extract_fenced_code(extracted["system_prompt.md"].decode("utf-8"))
    else:
        warnings.append("system_prompt.md missing — prompt not restored.")

    # -- Parse optional character payloads ----------------------------------- #

    character_a: dict | None = None
    if "char_a_payload.json" in extracted:
        try:
            character_a = json.loads(extracted["char_a_payload.json"].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            warnings.append("char_a_payload.json could not be parsed — skipped.")

    character_b: dict | None = None
    if "char_b_payload.json" in extracted:
        try:
            character_b = json.loads(extracted["char_b_payload.json"].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            warnings.append("char_b_payload.json could not be parsed — skipped.")

    # -- Parse optional game log --------------------------------------------- #

    game_log_entries: list[dict[str, str]] = []
    if "game_log.md" in extracted:
        game_log_entries = parse_game_log_md(extracted["game_log.md"].decode("utf-8"))

    return ChatImportResponse(
        folder_name=metadata.get("folder_name", "unknown"),
        metadata=metadata,
        character_a=character_a,
        character_b=character_b,
        system_prompt=system_prompt,
        game_log_entries=game_log_entries,
        model=metadata.get("model", "unknown"),
        temperature=metadata.get("temperature", 0.0),
        max_tokens=metadata.get("max_tokens", 128),
        seed=metadata.get("seed", -1),
        manifest_valid=True,
        files=sorted(extracted.keys()),
        warnings=warnings,
    )
