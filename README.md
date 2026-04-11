[![CI](https://github.com/pipe-works/pipeworks_axis_descriptor_lab/actions/workflows/ci.yml/badge.svg)](https://github.com/pipe-works/pipeworks_axis_descriptor_lab/actions/workflows/ci.yml) [![Documentation](https://readthedocs.org/projects/pipeworks-axis-descriptor-lab/badge/?version=latest)](https://pipeworks-axis-descriptor-lab.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/pipe-works/pipeworks_axis_descriptor_lab/branch/main/graph/badge.svg)](https://codecov.io/gh/pipe-works/pipeworks_axis_descriptor_lab) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# Axis Descriptor Lab

Axis Descriptor Lab is a FastAPI application with a browser-native frontend for
inspecting and experimenting with LLM output generated from deterministic
Pipe-Works axis payloads.

The lab is a **read-only inspection surface**: it consumes authoritative inputs
produced elsewhere and generates non-authoritative, ornamental output from them.
It never writes back to any canonical system.

## Core Concepts

### Axes

An *axis* is a named character dimension produced deterministically by the
Pipe-Works engine. Each axis has two fields:

- `label` — a short human-readable descriptor (e.g. `"resentful"`, `"weary"`)
- `score` — a normalised float in `[0, 1]`

A complete **axis payload** bundles a map of axes with a `policy_hash`
(SHA-256 of the active policy), a `seed` (the RNG seed that produced the
scores), and a `world_id`. This payload is serialised as JSON and sent verbatim
to the LLM as the user turn.

### Interpretive Provenance Chain (IPC)

Every generation is fingerprinted by a four-part hash chain:

| Hash | Covers |
|------|--------|
| `input_hash` | SHA-256 of the serialised axis payload |
| `system_prompt_hash` | SHA-256 of the system prompt text |
| `output_hash` | SHA-256 of the generated text |
| `ipc_id` | Combined chain hash (input + prompt + model + temperature + tokens + seed) |

Identical inputs always produce the same `ipc_id`. This makes prompt drift and
model drift auditable: two runs with the same `ipc_id` are provably identical;
a changed `ipc_id` shows exactly what changed.

### Authority Boundary

The boundary between authoritative and non-authoritative is strict:

- `pipeworks_mud_server` owns canonical runtime policy, prompt, and world state
- Deterministic payloads, hashes, and policy logic are authoritative
- Axis Lab is an inspection, comparison, and experimentation surface
- LLM output is ornamental and **never** becomes source of truth

## What The App Does

The UI exposes three main surfaces:

### Character Description

Input an axis payload (manually, via an example, or compiled from the mud
server), select an Ollama model and parameters, and generate a descriptive
paragraph. The panel displays the generated text alongside its full IPC hash
chain so identical runs can be detected and drift can be audited.

A baseline output can be pinned. Subsequent runs are compared against it using
two programmatic (non-LLM) analysis tools:

- **Signal Isolation** — tokenises, lemmatises, and stopword-filters both texts,
  then returns the set-difference of content lemmas as sorted `added`/`removed`
  word lists. Surfaces meaningful lexical pivots without structural noise.
- **Transformation Map** — sentence-aware clause alignment followed by
  token-level diffing. Returns ordered clause-replacement pairs, each annotated
  with structural micro-indicators (compression, expansion, embodiment shift,
  intensity change, etc.) derived from deterministic heuristics.

### Chat Translation

Translate OOC (out-of-character) player messages into IC (in-character) text.
Two modes are available and switchable at runtime without restarting the server:

- **Standalone / Offline** — calls a local Ollama instance directly
- **Server mode** — proxies to the mud server's canonical translation pipeline
  (requires authentication and a configured `MUD_SERVER_URL` or
  `MUD_SERVER_DEV_URL`)

Character sliders and a game-log panel are included. Sessions can be saved and
restored as zip packages.

### Pipeline Build

Inspect the mud server's policy inputs and compile deterministic image prompt
request bundles. Requires mud server authentication. Accepts species, gender,
axis scores, world context tags, and occupation signals; returns a compiled
prompt package with full provenance metadata from the canonical pipeline.

## Save, Export, and Import

All three surfaces support session persistence. A save package is a zip archive
containing:

- `metadata.json` — model, temperature, max_tokens, seed, folder name
- `payload.json` — the axis payload used
- `system_prompt.md` — the system prompt (fenced code block)
- `output.md` / `baseline.md` — generated texts (Character Description)
- `char_a_payload.json` / `char_b_payload.json` — character payloads (Chat)
- `game_log.md` — chat history (Chat)
- `manifest.json` — SHA-256 checksums for integrity validation

Packages can be exported as a download and re-imported to restore a complete
session.

## Local Assets And Canonical State

Local checked-in assets exist in two directories with different standing:

- `app/worlds/` — world-scoped assets (policies, zone data) used for local
  inspection and development. These are intentionally narrow in scope.
- `app/lab_only/` — explicitly non-canonical material (example payloads,
  experimental prompts, local policy bundles). Nothing here is authoritative.

Legacy mirror-era fallback behavior is not a compatibility target.

## Development

### Requirements

- Python `>=3.12`
- GitHub access for the `pipeworks-ipc` dependency (installed from a private
  GitHub repo — requires an SSH key or a `GITHUB_TOKEN` with repo read access)
- Ollama for standalone generation flows
- Mud server access when validating canonical server-mode behavior

### Install

```bash
pip install -e ".[dev]"
```

This pulls `pipeworks-ipc` directly from GitHub. If `pip` cannot reach the
repo, check your SSH config or set `GITHUB_TOKEN` before installing.

Docs tooling:

```bash
pip install -e ".[docs]"
```

### Local Environment

Copy the local dev env template:

```bash
cp .env.example .env
```

The local dev template defaults to:

- `APP_HOST=127.0.0.1`
- `APP_PORT=8242`
- Standalone Ollama mode unless `MUD_SERVER_URL` is set

### NLTK Data

The Signal Isolation and Transformation Map features require NLTK data
resources. Bootstrap them explicitly after installing:

```bash
python tools/bootstrap_nltk.py
```

The bootstrap path is explicit:

- If `NLTK_DATA` is set, that path is used
- Otherwise the fallback is repo-local `data/nltk_data`

Do not rely on NLTK's user-home defaults.

### Run Locally

Using the helper launcher:

```bash
python tools/dev_server.py
```

Or directly:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8242
```

Then open:

- `http://127.0.0.1:8242`

## Testing And Validation

Full test suite:

```bash
pytest
```

Coverage run:

```bash
pytest -v --cov --cov-report=term
```

Lint and format:

```bash
ruff check app tests
black app tests
```

Docs build:

```bash
make -C docs html
```

## Environment Variables

Local development uses `.env.example`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama base URL |
| `DEFAULT_MODEL` | `gemma2:2b` | Default standalone model |
| `APP_HOST` | `127.0.0.1` | Uvicorn bind host |
| `APP_PORT` | `8242` | Uvicorn bind port |
| `APP_RELOAD` | `1` for local dev launcher unless overridden | Uvicorn reload toggle |
| `MUD_SERVER_URL` | unset in local dev template | Canonical mud-server URL for configured server mode |
| `MUD_SERVER_DEV_URL` | `http://localhost:8000` when set | Development mud-server URL for runtime-selectable dev mode |
| `MUD_SERVER_TIMEOUT` | `120` | Timeout for mud-server proxy calls |
| `LAB_DEFAULT_WORLD_ID` | `pipeworks_web` | Default world selection |
| `AXIS_LAB_DATA_DIR` | repo-local `data/` unless explicitly overridden | Writable save/export root |
| `AXIS_LAB_LOGS_DIR` | repo-local `logs/` unless explicitly overridden | Writable log root |
| `NLTK_DATA` | repo-local `data/nltk_data` unless explicitly overridden | Explicit NLTK data root |

## API Reference

### SPA Shell

| Method | Path | Description |
|--------|------|-------------|
| `GET`, `HEAD` | `/` | Serve the SPA shell (Character Description preselected) |
| `GET` | `/pipeline-build` | Serve the SPA shell (Pipeline Build preselected) |

### Examples And Prompts

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/examples` | List available example payload names |
| `GET` | `/api/examples/{name}` | Return a named example payload as JSON |
| `GET` | `/api/prompts` | List available prompt names (filter by `purpose` query param) |
| `GET` | `/api/prompts/{name}` | Return a named prompt's text content |
| `GET` | `/api/system-prompt` | Return the default Character Description system prompt |

### Generation And Analysis

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/models` | List locally available Ollama models |
| `POST` | `/api/generate` | Generate descriptive text from an axis payload; returns text + IPC hashes |
| `POST` | `/api/log` | Append a run log entry (JSONL) to the configured log file |
| `POST` | `/api/relabel` | Recompute axis labels from policy score mappings |
| `POST` | `/api/analyze-delta` | Signal Isolation: content-lemma set-difference between two texts |
| `POST` | `/api/transformation-map` | Transformation Map: clause-level replacement pairs with micro-indicators |

### Save And Import

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/save` | Save a Character Description session as a zip package |
| `GET` | `/api/save/{name}/export` | Download a save package as a zip |
| `POST` | `/api/import` | Import a Character Description save package from a zip upload |
| `POST` | `/api/translate_chat` | OOC→IC translation for one or two characters |
| `POST` | `/api/save_chat` | Save a Chat Translation session as a zip package |
| `POST` | `/api/import_chat` | Import a Chat Translation save package from a zip upload |

### Mud Server Proxy (`/api/mud/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/mud/login` | Authenticate with the mud server; caches session token in memory |
| `POST` | `/api/mud/logout` | Clear the in-memory session token |
| `GET` | `/api/mud/session` | Return current auth status and selected world |
| `GET`, `POST` | `/api/mud/mode` | Get or switch the active runtime translation mode |
| `GET` | `/api/mud/worlds` | Proxy world list from the mud server |
| `GET` | `/api/mud/world-config/{id}` | Proxy world config from the mud server |
| `GET` | `/api/mud/world-prompts/{id}` | Proxy world prompt templates (falls back to policy API if legacy endpoint absent) |
| `GET` | `/api/mud/world-image-policy-bundle/{id}` | Proxy canonical image policy bundle metadata |
| `POST` | `/api/mud/compile-image-prompt` | Proxy canonical image prompt compilation |
| `POST` | `/api/mud/select-world` | Store the selected world ID in the active client session |

## Key Repo Areas

- `app/main.py` — FastAPI entrypoint, top-level route wiring, IPC hash computation
- `app/routes_chat.py` — chat translation and chat save routes
- `app/routes_mud.py` — mud-server proxy routes and pipeline bootstrap endpoints
- `app/routes_save.py` — system-prompt, save, and export routes
- `app/services/chat_translation.py` — OOC→IC translation orchestration
- `app/services/save_service.py` — Character Description save orchestration
- `app/signal_isolation.py` — NLP pipeline for content-word delta (Signal Isolation Layer)
- `app/transformation_map.py` — clause-level sentence alignment and diffing
- `app/micro_indicators.py` — structural pattern classifiers for Transformation Map rows
- `app/mud_server_client.py` — synchronous HTTP client, runtime mode state, connection pooling
- `app/chat_renderer.py` — synchronous Ollama HTTP wrapper
- `app/relabel_policy.py` — policy table and score-to-label mapping
- `app/save_package.py` — zip archive builder, manifest checksums, import extraction
- `app/save_formatting.py` — Markdown builders and folder-name generator
- `app/file_loaders.py` and `app/path_resolver.py` — local asset resolution
- `app/schema/` — Pydantic v2 request and response models
- `app/static/` — browser-native ES modules and CSS
- `app/templates/` — Jinja2 HTML shell template
- `app/worlds/` — world-scoped local assets
- `app/lab_only/` — non-canonical lab material
- `docs/` — Sphinx documentation source
- `tests/` — pytest suite
- `tools/` — dev server launcher, NLTK bootstrap

## Documentation

Published docs:

- <https://pipeworks-axis-descriptor-lab.readthedocs.io/en/latest/>

Useful pages:

- [IPC and Hashing Guide](https://pipeworks-axis-descriptor-lab.readthedocs.io/en/latest/guides/ipc-and-hashing.html)
- [API Reference](https://pipeworks-axis-descriptor-lab.readthedocs.io/en/latest/api/index.html)

## License

[GPL-3.0-or-later](LICENSE)
