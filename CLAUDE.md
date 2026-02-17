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

- **`app/main.py`** — FastAPI app with all routes. Sync handlers (not async); FastAPI runs them in a threadpool. Serves the Jinja2 template at `/` and all `/api/*` endpoints.
- **`app/schema.py`** — Pydantic v2 models: `AxisValue` (label + score 0.0–1.0), `AxisPayload` (dict of axes + policy_hash + seed + world_id), `GenerateRequest`, `GenerateResponse`, `LogEntry`.
- **`app/ollama_client.py`** — Synchronous HTTP wrapper around Ollama's `/api/generate` and `/api/tags` endpoints using httpx. 10s connect / 120s read timeout.

### Frontend (Vanilla JS)

- **`app/static/app.js`** — All interactive behavior: slider controls, JSON textarea sync, generate calls, baseline/diff comparison (word-level LCS diff algorithm), logging.
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
