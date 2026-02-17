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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.ollama_client import list_local_models, ollama_generate
from app.schema import AxisPayload, GenerateRequest, GenerateResponse, LogEntry

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

# Create logs directory if it doesn't exist (production deployments may not
# have committed the .gitkeep file).
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

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
