# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

The **Axis Descriptor Lab** is a single-user web tool for testing how small LLMs (via Ollama) produce non-authoritative descriptive text from deterministic axis payloads. It is part of the Pipe-Works ecosystem.

**Key principle**: The system (axes, scores, seeds) is **authoritative**. The LLM is **ornamental** — it produces flavor text only, never makes decisions, and its output is never trusted as ground truth.

## Commands

```bash
# Install (editable)
pip install -e .
pip install -e ".[dev]"    # includes pytest, ruff, black, mypy, bandit, pre-commit
pip install -e ".[docs]"   # includes sphinx, sphinx-rtd-theme, myst-parser

# Run server (requires Ollama running locally)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8242

# Tests
pytest                             # run all tests
pytest -v --cov --cov-report=term  # with coverage

# Lint
ruff check app/

# Docs
make -C docs html                  # build HTML docs to docs/_build/html/
```

## Architecture

The app is a FastAPI backend serving a vanilla JS single-page frontend. There are no build tools, no frontend frameworks, and no database.

### Backend (Python)

- **`app/main.py`** — FastAPI app: thin routing layer that orchestrates calls to domain modules. Sync handlers (not async); FastAPI runs them in a threadpool. Serves the Jinja2 template at `/` and all `/api/*` endpoints.
- **`app/schema.py`** — Pydantic v2 models: `AxisValue` (label + score 0.0–1.0), `AxisPayload` (dict of axes + policy_hash + seed + world_id), `GenerateRequest`, `GenerateResponse`, `LogEntry`, `DeltaRequest`, `DeltaResponse`, `IndicatorConfig`, `TransformationMapRow` (with `indicators`).
- **`app/hashing.py`** — IPC normalisation and hash utilities (payload, system prompt, output, composite IPC ID, typed `payload_hash` convenience wrapper).
- **`app/signal_isolation.py`** — NLP pipeline for the Signal Isolation Layer: tokenise (NLTK), lemmatise (WordNet), filter stopwords, compute content-word set delta between two texts. Requires NLTK data packages (punkt_tab, stopwords, wordnet) which are auto-downloaded on first run.
- **`app/ollama_client.py`** — Synchronous HTTP wrapper around Ollama's `/api/generate` and `/api/tags` endpoints using httpx. 10s connect / 120s read timeout.
- **`app/relabel_policy.py`** — Policy data table (`RELABEL_POLICY`) and `apply_relabel_policy()` function for score-to-label mapping across 11 axes.
- **`app/save_formatting.py`** — Pure formatting functions for the save system: `save_folder_name()`, `build_output_md()`, `build_baseline_md()`, `build_system_prompt_md()`. No I/O, no app dependencies.
- **`app/file_loaders.py`** — File-loading utilities: `load_default_prompt()`, `load_example()`, `load_prompt()`, `list_example_names()`, `list_prompt_names()`. Reads from `app/prompts/` and `app/examples/`.
- **`app/micro_indicators.py`** — Structural Learning Layer: 10 deterministic heuristic classifiers (`compression`, `expansion`, `embodiment shift`, `abstraction ↑`, `intensity ↑/↓`, `consolidation`, `fragmentation`, `modality shift`, `tone reframing`, `lexical pivot`) that label transformation-map rows. Uses NLTK for POS tagging/sentence segmentation and JSON lexicon data from `app/data/`. Configurable via `IndicatorConfig`.
- **`app/data/`** — JSON lexicon files for micro-indicators: `embodiment_v0_1.json`, `abstraction_v0_1.json`, `intensity_v0_1.json`.

### Frontend (Vanilla JS — ES Modules)

The frontend is split into 14 browser-native ES modules (`app/static/mod-*.js`). No bundler — `<script type="module">` loads the entry point and the browser resolves all imports.

- **`mod-init.js`** — Entry point; orchestrates startup (theme, tooltips, events, data loading).
- **`mod-state.js`** — Singleton state object + cached DOM refs (`state`, `dom`).
- **`mod-events.js`** — Thin coordinator calling all `wire*Events()` functions.
- **`mod-utils.js`** — Pure functions: `clamp`, `debounce`, `tokenise`, `lcsWordDiff`, `extractTransformationRows`, `cryptoRandomFloat`, `makePlaceholder`.
- **`mod-status.js`** — Status bar updates (`setStatus`).
- **`mod-sync.js`** — Bidirectional sync: `state.payload` ↔ JSON textarea ↔ slider panel. Form readers (`getModelName`, `resolveSeed`), model refresh.
- **`mod-loaders.js`** — Example and system prompt list/load from server.
- **`mod-generate.js`** — POST `/api/generate`, output rendering, meta table, diff trigger.
- **`mod-diff.js`** — Word-level LCS diff, signal isolation (NLP), transformation map (server-side with micro-indicator tags, client-side fallback).
- **`mod-axis-actions.js`** — Relabel (server policy), randomise, auto-label toggle.
- **`mod-persistence.js`** — Save, export zip, import zip, restore session, log.
- **`mod-indicator-modal.js`** — Indicator tooltip text + click-to-open modal with definitions, heuristics, examples, and docs link (standalone, no imports).
- **`mod-tooltip.js`** — JS-positioned tooltip system (standalone, no imports).
- **`mod-theme.js`** — Dark/light theme toggle with localStorage (standalone).

Supporting files:

- **`app/static/styles.css`** — Dark industrial theme with amber accents, CSS Grid 3-column layout.
- **`app/templates/index.html`** — SPA shell rendered by Jinja2; injects `default_model` and `available_models` at load time.

### Data Flow

1. User loads/edits an `AxisPayload` (JSON textarea or sliders)
2. Frontend POSTs to `/api/generate` with payload + model + temperature + max_tokens
3. Backend serializes the payload as the user prompt, loads the system prompt from `app/prompts/system_prompt_v01.txt`, calls Ollama
4. Response text is displayed; optionally stored as baseline for word-level diff comparison
5. Each run can be logged (append-only JSONL at `logs/run_log.jsonl`) with SHA-256 input hash for grouping

### Server-Side Policy (Relabeling)

The `/api/relabel` endpoint applies a policy table that maps score ranges to labels for known axes (e.g., age 0.25→"young", 0.75→"old"). This keeps label logic on the server, not hardcoded in JS.

## Environment Variables

Configured via `.env` (copy from `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `DEFAULT_MODEL` | `gemma2:2b` | Default model for generation |
| `APP_HOST` | `127.0.0.1` | Server bind address |
| `APP_PORT` | `8242` | Server port |

## CI/CD

- **CI**: GitHub Actions via org reusable workflows (`pipe-works/.github`). Runs on push to `main`/`develop`/`release-please--*` and PRs.
- **Release-please**: Automated versioning and changelog from conventional commits. Pushes to `main` trigger a release PR.
- **Branch protection**: `main` requires passing `ci / All Checks Passed` status check.
- **Codecov**: Coverage targets 50% project, 70% patch.

## Conventional Commits

Required for release-please. Format: `type(scope): description`

- `feat:` — new feature (bumps minor pre-1.0, patch pre-major)
- `fix:` — bug fix (bumps patch)
- `docs:`, `refactor:`, `chore:`, `ci:`, `test:`, `build:` — no version bump

## Code Style

- Python 3.12+, line length 100 (black + ruff)
- Type hints on Pydantic models and public APIs
- GPL-3.0 license
- Pre-commit hooks: `pre-commit install` then hooks run automatically on commit
- Determinism rules from the broader Pipe-Works ecosystem apply: use `random.Random(seed)`, never global `random.seed()`
