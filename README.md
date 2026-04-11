[![CI](https://github.com/pipe-works/pipeworks_axis_descriptor_lab/actions/workflows/ci.yml/badge.svg)](https://github.com/pipe-works/pipeworks_axis_descriptor_lab/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pipe-works/pipeworks_axis_descriptor_lab/branch/main/graph/badge.svg)](https://codecov.io/gh/pipe-works/pipeworks_axis_descriptor_lab)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# Axis Descriptor Lab

Axis Descriptor Lab is the non-authoritative inspection and experimentation
surface for deterministic PipeWorks axis payloads. It is a FastAPI application
with a browser-native frontend for generating descriptive text, comparing output
drift, translating chat, and inspecting selected mud-server-backed pipeline
inputs without writing back to canonical runtime state.

## PipeWorks Workspace

These repositories are designed to live inside a shared PipeWorks workspace
rooted at `/srv/work/pipeworks`.

- `repos/` contains source checkouts only.
- `venvs/` contains per-project virtual environments such as `pw-mud-server`.
- `runtime/` contains mutable runtime state such as databases, exports, session
  files, and caches.
- `logs/` contains service-owned log output when a project writes logs outside
  the process manager.
- `config/` contains workspace-level configuration files that should not be
  treated as source.
- `bin/` contains optional workspace helper scripts.
- `home/` is reserved for workspace-local user data when a project needs it.

Across the PipeWorks ecosphere, the rule is simple: keep source in `repos/`,
keep mutable state outside the repo checkout, and use explicit paths between
repos when one project depends on another.

## What This Repo Owns

This repository is the source of truth for:

- the browser UI and API for axis-payload inspection
- deterministic analysis helpers such as Signal Isolation and Transformation Map
- save/export/import of lab sessions
- local example payloads and lab-only assets
- mud-server proxy flows used by the lab UI

This repository does not own:

- canonical runtime policy, prompt, or world state
- authoritative activation state
- any path by which LLM output becomes runtime truth

## Main App Surfaces

### Character Description

Generate descriptive text from deterministic axis payloads and inspect the full
IPC hash chain for the run. Baselines can be pinned and compared using:

- Signal Isolation
- Transformation Map

### Chat Translation

Translate OOC player text into IC output in two modes:

- standalone mode against a local Ollama instance
- server mode proxied through `pipeworks_mud_server`

### Pipeline Build

Inspect mud-server policy inputs and compile deterministic image-prompt request
bundles through the canonical server pipeline.

## Ecosystem Dependencies

The lab can run in multiple levels of completeness:

- Character Description, standalone
  requires the lab plus a local Ollama instance
- Chat Translation, standalone
  requires the lab plus a local Ollama instance
- Chat Translation, server mode
  requires the lab, a reachable `pipeworks_mud_server`, and valid auth
- Pipeline Build
  requires the lab, a reachable `pipeworks_mud_server`, and canonical policy
  data available through that server

Shared library dependency:

- `pipeworks-ipc` for deterministic hashing and normalization

## Repository Layout

- `app/main.py` FastAPI entrypoint and top-level route wiring
- `app/routes_*.py` API route modules for save, chat, and mud-server proxying
- `app/services/` orchestration for save and chat behavior
- `app/signal_isolation.py` deterministic lexical-delta analysis
- `app/transformation_map.py` clause-level replacement analysis
- `app/micro_indicators.py` deterministic indicator classification
- `app/static/` browser-native frontend assets
- `app/templates/` HTML shell
- `app/worlds/` world-scoped local assets
- `app/lab_only/` explicitly non-canonical lab material
- `docs/` Sphinx documentation
- `tests/` pytest suite
- `tools/` developer helpers including NLTK bootstrap

## Quick Start

### Requirements

- Python `>=3.12`
- a PipeWorks workspace rooted at `/srv/work/pipeworks`
- Git access to the private `pipeworks-ipc` dependency referenced by
  `pyproject.toml`
- Ollama for standalone generation flows

### Install

```bash
python3 -m venv /srv/work/pipeworks/venvs/pw-axis-descriptor-lab
/srv/work/pipeworks/venvs/pw-axis-descriptor-lab/bin/pip install -e ".[dev]"
```

If you need docs tooling too:

```bash
/srv/work/pipeworks/venvs/pw-axis-descriptor-lab/bin/pip install -e ".[docs]"
```

### Local Environment

Create a local env file:

```bash
cp .env.example .env
```

For a workspace-backed run, the important paths are typically:

- `AXIS_LAB_DATA_DIR=/srv/work/pipeworks/runtime/axis-descriptor-lab`
- `AXIS_LAB_LOGS_DIR=/srv/work/pipeworks/logs/axis-descriptor-lab`
- `NLTK_DATA=/srv/work/pipeworks/runtime/axis-descriptor-lab/nltk_data`

If you want mud-server-backed behavior, also set:

- `MUD_SERVER_URL`
- or `MUD_SERVER_DEV_URL`

### Bootstrap NLTK Data

```bash
/srv/work/pipeworks/venvs/pw-axis-descriptor-lab/bin/python tools/bootstrap_nltk.py
```

### Run Locally

Using the helper launcher:

```bash
/srv/work/pipeworks/venvs/pw-axis-descriptor-lab/bin/python tools/dev_server.py
```

Or directly:

```bash
/srv/work/pipeworks/venvs/pw-axis-descriptor-lab/bin/uvicorn \
  app.main:app --reload --host 127.0.0.1 --port 8242
```

## Authority Boundary

The lab is intentionally read-only relative to canonical PipeWorks state.

- `pipeworks_mud_server` owns canonical runtime policy, prompts, and world state
- deterministic payloads and hashes are authoritative
- the lab may inspect, proxy, compare, and experiment around that state
- generated LLM output remains ornamental and non-authoritative

## Validation And Development

Run the main checks from the repo root:

```bash
/srv/work/pipeworks/venvs/pw-axis-descriptor-lab/bin/pytest
/srv/work/pipeworks/venvs/pw-axis-descriptor-lab/bin/ruff check app tests
/srv/work/pipeworks/venvs/pw-axis-descriptor-lab/bin/black --check app tests
```

Build the docs locally:

```bash
make -C docs html
```

## Documentation

Published docs:

- <https://pipeworks-axis-descriptor-lab.readthedocs.io/en/latest/>

Useful local doc areas include:

- `docs/`
- `app/worlds/README.md`
- `app/lab_only/README.md`

## License

[GPL-3.0-or-later](LICENSE)
