"""
app/main.py
-----------------------------------------------------------------------------
FastAPI application entrypoint for the Axis Descriptor Lab.

This module is a **thin routing layer** — each route handler orchestrates
calls to domain modules and returns the result.  All business logic lives
in dedicated modules:

Domain modules
~~~~~~~~~~~~~~
- ``app.hashing``          – IPC normalisation and SHA-256 hash functions.
- ``app.schema``           – Pydantic v2 request / response models.
- ``app.ollama_client``    – Synchronous HTTP wrapper around Ollama.
- ``app.signal_isolation`` – NLP pipeline for content-word delta.
- ``app.transformation_map`` – Clause-level sentence alignment and diffing.
- ``app.save_package``     – Manifest builder, zip archive, import/export.
- ``app.relabel_policy``   – Policy table and score-to-label mapping.
- ``app.save_formatting``  – Markdown builders and folder-name generator.
- ``app.file_loaders``     – Example and prompt file loading/listing.

Run with:
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8242

Endpoints
---------
GET  /                         → serves index.html
GET  /api/examples             → list of available example names
GET  /api/examples/{name}      → returns a single example JSON payload
GET  /api/prompts              → list of available prompt names
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

import io
import json
import os
import re
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.hashing import (
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
from app.ollama_client import OLLAMA_HOST, list_local_models, ollama_generate
from app.relabel_policy import apply_relabel_policy
from app.save_formatting import (
    build_baseline_md,
    build_output_md,
    build_system_prompt_md,
    save_folder_name,
)
from app.save_package import (
    build_manifest,
    create_zip_archive,
    extract_body_text,
    extract_fenced_code,
    validate_and_extract_zip,
    MAX_UPLOAD_SIZE,
)
from app.schema import (
    AxisPayload,
    DeltaRequest,
    DeltaResponse,
    GenerateRequest,
    GenerateResponse,
    ImportResponse,
    LogEntry,
    SaveRequest,
    SaveResponse,
    TransformationMapRequest,
    TransformationMapResponse,
    TransformationMapRow,
)
from app.signal_isolation import compute_delta
from app.transformation_map import compute_transformation_map

# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------

load_dotenv()

# Resolve paths relative to this file so the app works regardless of the
# working directory from which uvicorn is launched.
_HERE = Path(__file__).parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"
_LOGS_DIR = _HERE.parent / "logs"
_DATA_DIR = _HERE.parent / "data"

# Create runtime output directories if they don't exist (production
# deployments may not have committed .gitkeep files).
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE = _LOGS_DIR / "run_log.jsonl"

_DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemma2:2b")

# Read version from pyproject.toml (single source of truth).
_PYPROJECT = _HERE.parent / "pyproject.toml"
with open(_PYPROJECT, "rb") as _f:
    _APP_VERSION: str = tomllib.load(_f)["project"]["version"]

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
    available_models = list_local_models()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_model": _DEFAULT_MODEL,
            "available_models": available_models,
            "ollama_host": OLLAMA_HOST,
            "app_version": _APP_VERSION,
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
    name : Example stem, e.g. "example_a".

    Returns
    -------
    The raw example JSON object (validated loosely by Pydantic when the
    frontend loads it into the textarea).
    """
    return load_example(name)


@app.get("/api/prompts", summary="List available prompt names")
def list_prompts() -> list[str]:
    """
    Return a sorted list of prompt names (without the .txt extension) that
    are stored in app/prompts/.

    The frontend uses this to populate its prompt library dropdown, allowing
    users to browse and load different system prompt variants into the
    system prompt override textarea.
    """
    return list_prompt_names()


@app.get(
    "/api/prompts/{name}",
    response_class=PlainTextResponse,
    summary="Get a named prompt text",
)
def get_prompt(name: str) -> str:
    """
    Return the text content of a named prompt file as plain text.

    Uses ``PlainTextResponse`` to match the existing ``/api/system-prompt``
    endpoint pattern.  The frontend loads this into the system prompt
    override textarea.

    Parameters
    ----------
    name : Prompt stem, e.g. "system_prompt_v01".

    Returns
    -------
    The raw prompt text (plain text, not JSON).
    """
    return load_prompt(name)


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
    return list_local_models(host=host)


@app.post(
    "/api/generate",
    response_model=GenerateResponse,
    summary="Generate a descriptive paragraph from an axis payload",
)
def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Core endpoint: serialise the axis payload to JSON, attach the system
    prompt, call Ollama, and return the generated paragraph.

    The system prompt is taken from the request body if provided, otherwise
    the default prompt (app/prompts/system_prompt_v01.txt) is loaded from
    disk.  This lets the frontend experiment with custom prompts without
    restarting the server.

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
        text, usage = ollama_generate(
            model=req.model,
            system_prompt=system_prompt,
            user_json_str=user_json_str,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            seed=req.payload.seed,
            host=req.ollama_host,
        )
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
    summary="Compute clause-level replacement pairs between baseline and current text",
)
def transformation_map(req: TransformationMapRequest) -> TransformationMapResponse:
    """
    Transformation Map endpoint.

    Takes two text strings (baseline A and current B), runs sentence-aware
    alignment followed by token-level diffing, and returns clause-level
    replacement pairs.

    This fills the gap between word-level diff (too granular) and
    content-word delta (structure-blind) by showing *what chunk of text
    was replaced by what chunk* at the clause scale.

    The LLM is not involved — this is pure programmatic text analysis.

    Parameters
    ----------
    req : TransformationMapRequest
        Contains ``baseline_text`` and ``current_text``.

    Returns
    -------
    TransformationMapResponse
        Ordered list of ``TransformationMapRow`` replacement pairs.
    """
    rows = compute_transformation_map(
        req.baseline_text, req.current_text, include_all=req.include_all
    )
    return TransformationMapResponse(rows=[TransformationMapRow(**row) for row in rows])


# -----------------------------------------------------------------------------
# Save + system-prompt routes
# -----------------------------------------------------------------------------


@app.get(
    "/api/system-prompt",
    response_class=PlainTextResponse,
    summary="Return the default system prompt text",
)
def get_system_prompt() -> str:
    """
    Return the content of ``system_prompt_v01.txt`` as plain text.

    This allows the frontend to resolve the effective system prompt when no
    custom override is set, ensuring saved files always contain the complete
    prompt text rather than a placeholder.

    Returns
    -------
    str : The default system prompt as plain text.
    """
    return load_default_prompt()


@app.post(
    "/api/save",
    response_model=SaveResponse,
    summary="Save current session state to a timestamped subfolder under data/",
)
def save_run(req: SaveRequest) -> SaveResponse:
    """
    Persist all session state to individual files inside a new subfolder of
    ``data/``.

    The folder is named with a UTC datetime stamp and the first 8 characters
    of the payload's SHA-256 hash, guaranteeing practical uniqueness even if
    the user clicks Save multiple times within the same second (different
    payload → different hash suffix).

    Files written
    -------------
    metadata.json     – Model name, temperature, max_tokens, seed, timestamp,
                        input_hash, world_id, policy_hash, and axis count.
    payload.json      – The full AxisPayload as pretty-printed JSON.
    system_prompt.md  – The system prompt used for the generation.
    output.md         – The generated text (only if ``output`` is not None).
    baseline.md       – The baseline text (only if ``baseline`` is not None).
    delta.json        – Content-word delta between baseline and output (only
                        if both ``output`` and ``baseline`` are not None).
                        Contains added/removed content lemmas from the signal
                        isolation pipeline.  This is a derived analysis — it
                        does not affect IPC hashes.

    Parameters
    ----------
    req : SaveRequest – complete session snapshot from the frontend.

    Returns
    -------
    SaveResponse with the folder name, file list, hash, and timestamp.

    Raises
    ------
    HTTPException(500) if file I/O fails.
    """
    now = datetime.now(timezone.utc)
    input_hash = payload_hash(req.payload)
    folder_name = save_folder_name(now, input_hash)
    save_dir = _DATA_DIR / folder_name

    # -- Compute IPC hashes ------------------------------------------------
    # The system prompt hash is always available (system_prompt is required
    # on SaveRequest).  The output hash and IPC ID are only meaningful when
    # output text exists — without output the provenance chain is incomplete.
    sp_hash = compute_system_prompt_hash(req.system_prompt)
    out_hash = compute_output_hash(req.output) if req.output is not None else None
    ipc: str | None = None
    if req.output is not None:
        ipc = compute_ipc_id(
            input_hash=input_hash,
            system_prompt_hash=sp_hash,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            seed=req.payload.seed,
        )

    # Create the subfolder.  exist_ok=True in case rapid saves produce the
    # same folder name (same second + same payload hash).
    save_dir.mkdir(parents=True, exist_ok=True)

    files_written: list[str] = []

    try:
        # -- payload.json ------------------------------------------------ #
        # The full axis payload as pretty-printed JSON, identical to the
        # JSON textarea content in the UI.
        (save_dir / "payload.json").write_text(
            json.dumps(req.payload.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files_written.append("payload.json")

        # -- system_prompt.md -------------------------------------------- #
        # Always written — the prompt is required by SaveRequest.
        system_prompt_md = build_system_prompt_md(req.system_prompt, folder_name)
        (save_dir / "system_prompt.md").write_text(system_prompt_md, encoding="utf-8")
        files_written.append("system_prompt.md")

        # -- output.md (conditional) ------------------------------------- #
        # Only written when the user has generated text.  Omitting the
        # file (rather than writing an empty one) makes the save folder
        # self-documenting: if output.md is absent, no generation occurred.
        if req.output is not None:
            output_md = build_output_md(
                req.output,
                model=req.model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                seed=req.payload.seed,
                timestamp=now,
                input_hash=input_hash,
                system_prompt_hash=sp_hash,
                ipc_id=ipc,
            )
            (save_dir / "output.md").write_text(output_md, encoding="utf-8")
            files_written.append("output.md")

        # -- baseline.md (conditional) ----------------------------------- #
        # Only written when a baseline was stored via "Set as A".
        if req.baseline is not None:
            baseline_md = build_baseline_md(req.baseline, folder_name)
            (save_dir / "baseline.md").write_text(baseline_md, encoding="utf-8")
            files_written.append("baseline.md")

        # -- delta.json (conditional) ------------------------------------ #
        # Only written when both output and baseline exist, since the
        # content-word delta requires two texts to compare.
        #
        # This is a *derived analysis* — the delta is a deterministic
        # function of the baseline and output texts, both of which are
        # already saved above.  It does NOT affect IPC hashes or any
        # provenance computation; it simply persists the signal isolation
        # results for convenient reference without re-running the pipeline.
        if req.output is not None and req.baseline is not None:
            removed, added = compute_delta(req.baseline, req.output)
            delta_data = {
                "removed": removed,
                "added": added,
                "removed_count": len(removed),
                "added_count": len(added),
            }
            (save_dir / "delta.json").write_text(
                json.dumps(delta_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            files_written.append("delta.json")

        # -- transformation_map.json (conditional) ---------------------- #
        # Written when the frontend provides clause-level transformation
        # map rows (computed client-side from the word-level LCS diff).
        # Like delta.json, this is a derived analysis that does not affect
        # IPC hashes or provenance computation.
        if req.transformation_map is not None and len(req.transformation_map) > 0:
            tmap_data = {
                "rows": [row.model_dump() for row in req.transformation_map],
                "row_count": len(req.transformation_map),
            }
            (save_dir / "transformation_map.json").write_text(
                json.dumps(tmap_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            files_written.append("transformation_map.json")

        # -- metadata.json (written LAST) -------------------------------- #
        # metadata.json is written after all other files so that the
        # manifest can include SHA-256 checksums of every file.  The
        # manifest makes the save package self-describing and verifiable.
        metadata: dict[str, object] = {
            "folder_name": folder_name,
            "timestamp": now.isoformat(),
            "input_hash": input_hash,
            "system_prompt_hash": sp_hash,
            "output_hash": out_hash,
            "ipc_id": ipc,
            "model": req.model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "seed": req.payload.seed,
            "world_id": req.payload.world_id,
            "policy_hash": req.payload.policy_hash,
            "axis_count": len(req.payload.axes),
            "diff_change_pct": req.diff_change_pct,
        }

        # Build the manifest from all files written so far (before
        # metadata.json itself exists).  metadata.json gets sha256=null
        # in the manifest because it cannot hash its own content.
        manifest = build_manifest(save_dir, files_written)
        metadata["manifest"] = manifest

        (save_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files_written.append("metadata.json")

    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write save files to {save_dir}: {exc}",
        ) from exc

    return SaveResponse(
        folder_name=folder_name,
        files=sorted(files_written),
        input_hash=input_hash,
        timestamp=now.isoformat(),
        system_prompt_hash=sp_hash,
        output_hash=out_hash,
        ipc_id=ipc,
    )


# -----------------------------------------------------------------------------
# GET /api/save/{folder_name}/export
# -----------------------------------------------------------------------------

# Regex pattern for valid save folder names (YYYYMMDD_HHMMSS_<8hex>).
# Used as a security check to prevent path traversal via crafted folder names.
_FOLDER_NAME_PATTERN = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{8}$")


@app.get(
    "/api/save/{folder_name}/export",
    summary="Download a save package as a zip file",
)
def export_save(folder_name: str) -> StreamingResponse:
    """
    Stream a save folder as a compressed zip archive for download.

    The endpoint validates the folder name against a strict pattern to
    prevent path-traversal attacks, then bundles all known save-package
    files into a zip using ``save_package.create_zip_archive()``.

    Parameters
    ----------
    folder_name : The save folder name (e.g. '20260219_101437_5d628967').
                  Must match the ``YYYYMMDD_HHMMSS_<8hex>`` pattern.

    Returns
    -------
    StreamingResponse : A zip file with ``Content-Disposition: attachment``.

    Raises
    ------
    HTTPException(400) : If the folder name doesn't match the expected pattern.
    HTTPException(404) : If the folder does not exist under ``data/``.
    """
    # Security: validate folder name format to prevent path traversal.
    if not _FOLDER_NAME_PATTERN.match(folder_name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid folder name format: '{folder_name}'.",
        )

    save_dir = _DATA_DIR / folder_name
    if not save_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Save folder not found: '{folder_name}'.",
        )

    zip_bytes = create_zip_archive(save_dir)

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{folder_name}.zip"',
        },
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
