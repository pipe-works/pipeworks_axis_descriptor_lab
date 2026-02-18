"""
app/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application entrypoint for the Axis Descriptor Lab.

Run with:
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Endpoints
─────────
GET  /                      → serves index.html
GET  /api/examples          → list of available example names
GET  /api/examples/{name}   → returns a single example JSON payload
GET  /api/models            → returns locally available Ollama models
POST /api/generate          → send axis payload to Ollama, return description
POST /api/log               → persist a run log entry to logs/run_log.jsonl
POST /api/relabel           → (optional) recompute labels from policy rules
GET  /api/system-prompt     → return the default system prompt as plain text
POST /api/save              → save session state to a timestamped data/ subfolder

Architecture notes
──────────────────
• All blocking I/O (file reads, Ollama HTTP calls) lives in regular `def`
  route handlers.  FastAPI automatically runs those in a threadpool so the
  async event loop is never blocked.
• Static files are served by Starlette's StaticFiles middleware.
• Jinja2Templates renders index.html (single page – the JS takes over).
• A simple JSONL log file under logs/ provides the Pipe-Works audit trail
  without any database dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.ollama_client import list_local_models, ollama_generate
from app.schema import (
    AxisPayload,
    GenerateRequest,
    GenerateResponse,
    LogEntry,
    SaveRequest,
    SaveResponse,
)

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

# Resolve paths relative to this file so the app works regardless of the
# working directory from which uvicorn is launched.
_HERE = Path(__file__).parent
_PROMPTS_DIR = _HERE / "prompts"
_EXAMPLES_DIR = _HERE / "examples"
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

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app + middleware
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Axis Descriptor Lab",
    description=(
        "Tiny web tool for testing how small LLMs (via Ollama) produce "
        "non-authoritative descriptive text from a deterministic axis payload."
    ),
    version="0.1.0",
)

# Serve everything under /static/ directly from the filesystem.
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Jinja2 for the single HTML page.
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_default_prompt() -> str:
    """
    Read the default system prompt from disk.

    Returns the text of app/prompts/system_prompt_v01.txt.

    Raises
    ──────
    HTTPException(500) if the file is missing.
    """
    prompt_path = _PROMPTS_DIR / "system_prompt_v01.txt"
    if not prompt_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Default system prompt not found at {prompt_path}",
        )
    return prompt_path.read_text(encoding="utf-8").strip()


def _payload_hash(payload: AxisPayload) -> str:
    """
    Produce a stable SHA-256 hex digest for an AxisPayload.

    The payload is serialised with sorted keys so the hash is deterministic
    regardless of insertion order.

    Parameters
    ──────────
    payload : The AxisPayload to hash.

    Returns
    ───────
    str : 64-character lowercase hex digest.
    """
    # model_dump() returns a plain dict; json.dumps with sort_keys ensures
    # the serialisation is canonical.
    canonical = json.dumps(payload.model_dump(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_example(name: str) -> dict:
    """
    Load and parse a named example JSON file from app/examples/.

    Parameters
    ──────────
    name : Bare filename without extension (e.g. "example_a").

    Returns
    ───────
    dict : Parsed JSON object.

    Raises
    ──────
    HTTPException(404) if the file doesn't exist.
    HTTPException(500) if the file contains invalid JSON.
    """
    path = _EXAMPLES_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Example '{name}' not found.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"Example '{name}' contains invalid JSON: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


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
        },
    )


@app.get("/api/examples", summary="List available example names")
def list_examples() -> list[str]:
    """
    Return a sorted list of example names (without the .json extension) that
    are stored in app/examples/.

    The frontend uses this to populate its example dropdown.
    """
    names = sorted(p.stem for p in _EXAMPLES_DIR.glob("*.json"))
    return names


@app.get("/api/examples/{name}", summary="Get a named example payload")
def get_example(name: str) -> dict:
    """
    Return the parsed JSON for a named example.

    Parameters
    ──────────
    name : Example stem, e.g. "example_a".

    Returns
    ───────
    The raw example JSON object (validated loosely by Pydantic when the
    frontend loads it into the textarea).
    """
    return _load_example(name)


@app.get("/api/models", summary="List locally available Ollama models")
def get_models() -> list[str]:
    """
    Query the local Ollama instance and return all model names it has pulled.

    Returns an empty list if Ollama is unreachable, allowing the frontend to
    fall back to a manual text input.
    """
    return list_local_models()


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
    ──────────
    req : GenerateRequest – full request body (payload + model settings).

    Returns
    ───────
    GenerateResponse containing the generated text and metadata.

    Raises
    ──────
    HTTPException(502) if Ollama returns an HTTP error.
    HTTPException(504) if the Ollama request times out.
    HTTPException(500) for any other unexpected error.
    """
    system_prompt = req.system_prompt or _load_default_prompt()

    # Serialise the payload as pretty-printed JSON – this is the "user turn"
    # the LLM receives.  The prompt instructs it not to reference JSON by name.
    user_json_str = json.dumps(req.payload.model_dump(), ensure_ascii=False, indent=2)

    try:
        text, usage = ollama_generate(
            model=req.model,
            system_prompt=system_prompt,
            user_json_str=user_json_str,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
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

    return GenerateResponse(
        text=text,
        model=req.model,
        temperature=req.temperature,
        usage=usage,
    )


@app.post("/api/log", response_model=LogEntry, summary="Persist a run log entry")
def log_run(
    payload: AxisPayload,
    output: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> LogEntry:
    """
    Append a structured log entry to logs/run_log.jsonl.

    Each line in the JSONL file is one complete LogEntry serialised as compact
    JSON.  The file can be opened in any JSONL-aware tool (jq, pandas, etc.)
    for drift analysis.

    Parameters
    ──────────
    payload     : The AxisPayload used in the run.
    output      : The LLM-generated text.
    model       : Ollama model identifier.
    temperature : Sampling temperature used.
    max_tokens  : Token budget used.

    Returns
    ───────
    The complete LogEntry that was persisted.
    """
    entry = LogEntry(
        input_hash=_payload_hash(payload),
        payload=payload,
        output=output,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timestamp=datetime.now(timezone.utc).isoformat(),
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

    The mapping is intentionally simple and Pipe-Works-flavoured – it is NOT
    a substitute for a real policy engine.  It is here to make the score
    sliders meaningful and to show how label changes propagate to LLM output.

    Parameters
    ──────────
    payload : Current AxisPayload (axes with scores, possibly stale labels).

    Returns
    ───────
    Updated AxisPayload with labels recomputed from scores.
    """
    # Policy table: axis_name → list of (upper_bound_exclusive, label).
    # Thresholds are checked in order; the first match wins.
    # A final entry with upper_bound = 1.01 acts as the catch-all.
    _POLICY: dict[str, list[tuple[float, str]]] = {
        "age": [
            (0.25, "young"),
            (0.5, "middle-aged"),
            (0.75, "old"),
            (1.01, "ancient"),
        ],
        "demeanor": [
            (0.2, "cordial"),
            (0.4, "guarded"),
            (0.6, "resentful"),
            (0.8, "hostile"),
            (1.01, "menacing"),
        ],
        "dependency": [
            (0.33, "dispensable"),
            (0.66, "necessary"),
            (1.01, "indispensable"),
        ],
        "facial_signal": [
            (0.3, "open"),
            (0.6, "asymmetrical"),
            (1.01, "closed"),
        ],
        "health": [
            (0.25, "vigorous"),
            (0.5, "weary"),
            (0.75, "ailing"),
            (1.01, "failing"),
        ],
        "legitimacy": [
            (0.25, "unchallenged"),
            (0.5, "tolerated"),
            (0.65, "questioned"),
            (0.8, "contested"),
            (1.01, "illegitimate"),
        ],
        "moral_load": [
            (0.3, "clear"),
            (0.6, "conflicted"),
            (1.01, "burdened"),
        ],
        "physique": [
            (0.3, "gaunt"),
            (0.45, "lean"),
            (0.55, "stocky"),
            (0.7, "hunched"),
            (1.01, "imposing"),
        ],
        "risk_exposure": [
            (0.33, "sheltered"),
            (0.66, "hazardous"),
            (1.01, "perilous"),
        ],
        "visibility": [
            (0.33, "obscure"),
            (0.66, "routine"),
            (1.01, "prominent"),
        ],
        "wealth": [
            (0.25, "destitute"),
            (0.45, "threadbare"),
            (0.55, "well-kept"),
            (0.75, "comfortable"),
            (1.01, "affluent"),
        ],
    }

    # Build an updated axes dict with recomputed labels where policy exists.
    updated_axes = {}
    for axis_name, axis_val in payload.axes.items():
        if axis_name in _POLICY:
            new_label = axis_val.label  # default: keep existing
            for upper_bound, label in _POLICY[axis_name]:
                if axis_val.score < upper_bound:
                    new_label = label
                    break
            # Create a new AxisValue with the updated label, same score.
            updated_axes[axis_name] = axis_val.model_copy(update={"label": new_label})
        else:
            updated_axes[axis_name] = axis_val

    # Return a new payload with updated axes; all other fields unchanged.
    return payload.model_copy(update={"axes": updated_axes})


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save_folder_name(timestamp: datetime, input_hash: str) -> str:
    """
    Produce a unique folder name for a save operation.

    Format: ``YYYYMMDD_HHMMSS_<8-char-hash-prefix>``

    Example: ``20260218_143022_d845cdcf``

    Parameters
    ──────────
    timestamp  : UTC datetime of the save (passed in so the folder name stays
                 consistent with the ``metadata.json`` timestamp).
    input_hash : Full 64-char SHA-256 hex digest of the AxisPayload.

    Returns
    ───────
    str : Folder name safe for all major filesystems (no spaces, no special
          characters beyond underscores).
    """
    date_part = timestamp.strftime("%Y%m%d_%H%M%S")
    hash_part = input_hash[:8]
    return f"{date_part}_{hash_part}"


def _build_output_md(text: str, req: SaveRequest, now: datetime, input_hash: str) -> str:
    """
    Format the generated LLM output as a Markdown document.

    Includes an HTML-comment provenance header (model, temperature, seed,
    hash) so the file is self-documenting when opened in any Markdown viewer.

    Parameters
    ──────────
    text       : The raw LLM-generated text.
    req        : The full SaveRequest (for metadata fields).
    now        : UTC datetime of the save (for the provenance header).
    input_hash : SHA-256 of the payload.

    Returns
    ───────
    str : Markdown string ready to write to disk.
    """
    lines = [
        "# Output",
        "",
        "<!-- Axis Descriptor Lab – generated output -->",
        f"<!-- saved: {now.isoformat()} -->",
        f"<!-- model: {req.model} | temp: {req.temperature} " f"| max_tokens: {req.max_tokens} -->",
        f"<!-- seed: {req.payload.seed} | input_hash: {input_hash[:16]}... -->",
        "",
        text,
        "",
    ]
    return "\n".join(lines)


def _build_baseline_md(text: str, folder_name: str) -> str:
    """
    Format the stored baseline text as a Markdown document.

    Parameters
    ──────────
    text        : The baseline text (state.baseline from the frontend).
    folder_name : Save folder name (used in the provenance comment).

    Returns
    ───────
    str : Markdown string ready to write to disk.
    """
    lines = [
        "# Baseline (A)",
        "",
        f"<!-- Axis Descriptor Lab – baseline text for save {folder_name} -->",
        "",
        text,
        "",
    ]
    return "\n".join(lines)


def _build_system_prompt_md(prompt_text: str, folder_name: str) -> str:
    """
    Format the system prompt as a Markdown document with a fenced code block.

    Wrapping in a fenced code block preserves all whitespace and makes the
    prompt clearly machine-readable when opened in a Markdown viewer.

    Parameters
    ──────────
    prompt_text : The system prompt string (may be multi-line).
    folder_name : Save folder name (for the provenance comment).

    Returns
    ───────
    str : Markdown string ready to write to disk.
    """
    lines = [
        "# System Prompt",
        "",
        f"<!-- Axis Descriptor Lab – system prompt for save {folder_name} -->",
        "",
        "```text",
        prompt_text,
        "```",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Save + system-prompt routes
# ─────────────────────────────────────────────────────────────────────────────


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
    ───────
    str : The default system prompt as plain text.
    """
    return _load_default_prompt()


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
    ─────────────
    metadata.json     – Model name, temperature, max_tokens, seed, timestamp,
                        input_hash, world_id, policy_hash, and axis count.
    payload.json      – The full AxisPayload as pretty-printed JSON.
    system_prompt.md  – The system prompt used for the generation.
    output.md         – The generated text (only if ``output`` is not None).
    baseline.md       – The baseline text (only if ``baseline`` is not None).

    Parameters
    ──────────
    req : SaveRequest – complete session snapshot from the frontend.

    Returns
    ───────
    SaveResponse with the folder name, file list, hash, and timestamp.

    Raises
    ──────
    HTTPException(500) if file I/O fails.
    """
    now = datetime.now(timezone.utc)
    input_hash = _payload_hash(req.payload)
    folder_name = _save_folder_name(now, input_hash)
    save_dir = _DATA_DIR / folder_name

    # Create the subfolder.  exist_ok=True in case rapid saves produce the
    # same folder name (same second + same payload hash).
    save_dir.mkdir(parents=True, exist_ok=True)

    files_written: list[str] = []

    try:
        # ── metadata.json ─────────────────────────────────────────────── #
        # A flat dict of provenance fields for quick indexing across many
        # saves without parsing the full payload.
        metadata = {
            "folder_name": folder_name,
            "timestamp": now.isoformat(),
            "input_hash": input_hash,
            "model": req.model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "seed": req.payload.seed,
            "world_id": req.payload.world_id,
            "policy_hash": req.payload.policy_hash,
            "axis_count": len(req.payload.axes),
        }
        (save_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files_written.append("metadata.json")

        # ── payload.json ──────────────────────────────────────────────── #
        # The full axis payload as pretty-printed JSON, identical to the
        # JSON textarea content in the UI.
        (save_dir / "payload.json").write_text(
            json.dumps(req.payload.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files_written.append("payload.json")

        # ── system_prompt.md ──────────────────────────────────────────── #
        # Always written — the prompt is required by SaveRequest.
        system_prompt_md = _build_system_prompt_md(req.system_prompt, folder_name)
        (save_dir / "system_prompt.md").write_text(system_prompt_md, encoding="utf-8")
        files_written.append("system_prompt.md")

        # ── output.md (conditional) ───────────────────────────────────── #
        # Only written when the user has generated text.  Omitting the
        # file (rather than writing an empty one) makes the save folder
        # self-documenting: if output.md is absent, no generation occurred.
        if req.output is not None:
            output_md = _build_output_md(req.output, req, now, input_hash)
            (save_dir / "output.md").write_text(output_md, encoding="utf-8")
            files_written.append("output.md")

        # ── baseline.md (conditional) ─────────────────────────────────── #
        # Only written when a baseline was stored via "Set as A".
        if req.baseline is not None:
            baseline_md = _build_baseline_md(req.baseline, folder_name)
            (save_dir / "baseline.md").write_text(baseline_md, encoding="utf-8")
            files_written.append("baseline.md")

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
    )
