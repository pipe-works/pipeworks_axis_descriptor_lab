[![CI](https://github.com/pipe-works/pipeworks_axis_descriptor_lab/actions/workflows/ci.yml/badge.svg)](https://github.com/pipe-works/pipeworks_axis_descriptor_lab/actions/workflows/ci.yml) [![Documentation](https://readthedocs.org/projects/pipeworks-axis-descriptor-lab/badge/?version=latest)](https://pipeworks-axis-descriptor-lab.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/pipe-works/pipeworks_axis_descriptor_lab/branch/main/graph/badge.svg)](https://codecov.io/gh/pipe-works/pipeworks_axis_descriptor_lab) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# Axis Descriptor Lab

Axis Descriptor Lab is a FastAPI application with a browser-native JavaScript
frontend for testing and inspecting **non-authoritative** LLM output around
deterministic PipeWorks inputs.

The core rule is simple:

- deterministic payloads, hashes, policy logic, and mud-server-owned API state
  are authoritative
- LLM output is ornamental and must never become the source of truth

## Current Role

This repo now lives in the real PipeWorks workspace on `luminal.local`:

- workspace root: `/srv/work/pipeworks`
- repo path: `/srv/work/pipeworks/repos/pipeworks_axis_descriptor_lab`

Its Luminal host posture is currently being formalized in:

- `/home/aapark/dotfiles/docs/moc/luminal_pipeworks_axis_descriptor_lab_host_preparation.md`

Current direction:

- the repo should remain venv-backed for active development
- the repo is also intended to become a deliberate host-managed browser surface
- old workstation-era mirror-file behavior is not a compatibility target
- when canonical truth is in question, `pipeworks_mud_server` APIs should win
  over duplicated local copies

## What The Lab Does

The current UI has three main surfaces:

- `Character Description`
  Generate descriptive text from deterministic axis payloads and inspect the
  resulting differences, hashes, and provenance.
- `Chat Translation`
  Translate OOC player messages into IC speech either against local Ollama or
  through canonical mud-server-backed server mode.
- `Pipeline Build`
  Inspect canonical mud-server policy inputs and compile deterministic image
  prompt request bundles.

## Authority Boundary

Use this mental model when working in the repo:

1. `pipeworks_mud_server` owns canonical runtime policy, prompt, and world
   state.
2. Axis Lab may inspect, proxy, compare, compile, or experiment around that
   state.
3. Local lab-only assets are acceptable only when their role is explicit.
4. LLM output is never canonical.

Practical consequences:

- do not preserve legacy local mirrors just because they are convenient
- do not expand fallback file-resolution behavior without a clear supported use
  case
- if a local asset remains, document whether it is world-scoped or lab-only
- transitional mirror-era fallback paths are not part of the supported model

## Luminal Notes

This repo should now be read as part of a shared host environment, not only as
a workstation-local application.

Important implications:

- the supported host-managed posture is to write mutable state under
  `/srv/work/pipeworks/runtime/axis-descriptor-lab` and
  `/srv/work/pipeworks/logs/axis-descriptor-lab`
- repo-local `data/` and `logs/` are now fallback locations for local
  development when the host-managed paths are not available or not writable
- hostname, nginx, `systemd`, and runtime-path decisions should align with the
  active Luminal MOC and host docs
- the presence of a FastAPI server does not by itself define the final service
  topology

Current first-pass service identity:

- hostname: `https://descriptors.pipeworks.luminal.local/`
- backend bind: `127.0.0.1:8050`
- systemd unit: `pipeworks-axis-descriptor-lab.service`
- env file: `/etc/pipeworks/axis-descriptor-lab/axis-descriptor-lab.env`
- checked-in deploy templates:
  - `deploy/systemd/pipeworks-axis-descriptor-lab.service`
  - `deploy/nginx/descriptors.pipeworks.luminal.local`
  - `deploy/env/axis-descriptor-lab.env.example`

## Development

### Requirements

- Python `>=3.12` per `pyproject.toml`
- Ollama for local standalone generation workflows
- access to `pipeworks_mud_server` when testing canonical server-mode behavior

### Install

```bash
pip install -e .
pip install -e ".[dev]"
```

### Run Locally

```bash
cp .env.example .env
python tools/bootstrap_nltk.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8242
```

Or:

```bash
python tools/dev_server.py
```

Then open `http://127.0.0.1:8242`.

### Common Commands

```bash
pytest
pytest -v --cov --cov-report=term

ruff check app tests
black app tests

pip install -e ".[docs]"
make -C docs html
```

## Environment

Configured through `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama base URL |
| `DEFAULT_MODEL` | `gemma2:2b` | Default standalone model |
| `APP_HOST` | `127.0.0.1` | Uvicorn bind host |
| `APP_PORT` | `8242` | Uvicorn bind port |
| `MUD_SERVER_URL` | _(unset)_ | Canonical mud-server URL for configured server mode |
| `MUD_SERVER_DEV_URL` | `http://localhost:8000` | Development mud-server URL for runtime-selectable dev mode |
| `MUD_SERVER_TIMEOUT` | `120` | Timeout in seconds for mud-server proxy calls |
| `AXIS_LAB_DATA_DIR` | `/srv/work/pipeworks/runtime/axis-descriptor-lab` when writable, else repo-local `data/` | Writable save/export root |
| `AXIS_LAB_LOGS_DIR` | `/srv/work/pipeworks/logs/axis-descriptor-lab` when writable, else repo-local `logs/` | Writable log root |

Analysis features also require NLTK data resources. Bootstrap them explicitly
inside the active venv with:

```bash
python tools/bootstrap_nltk.py
```

## Key Files

- `app/main.py`
  FastAPI bootstrap and top-level route wiring.
- `app/routes_mud.py`
  Mud-server proxy routes and pipeline bootstrap endpoints.
- `app/mud_server_client.py`
  Shared mud-server client and runtime mode handling.
- `app/services/`
  Save and chat orchestration helpers.
- `app/file_loaders.py` and `app/path_resolver.py`
  Local asset resolution logic. Supported local reads now mean only
  world-scoped and explicitly lab-only assets; mirror-era fallback paths are
  being removed rather than preserved.
- `app/static/`
  Browser-native ES modules and CSS. No bundler.
- `docs/`
  Sphinx docs and narrative guides.
- `tests/`
  Pytest coverage for routes, domain logic, mud-server behavior, and save
  flows.

## Documentation

Published docs:

- <https://pipeworks-axis-descriptor-lab.readthedocs.io/en/latest/>

Useful pages:

- [IPC and Hashing Guide](https://pipeworks-axis-descriptor-lab.readthedocs.io/en/latest/guides/ipc-and-hashing.html)
- [API Reference](https://pipeworks-axis-descriptor-lab.readthedocs.io/en/latest/api/index.html)

## Contributing

Before changing behavior:

- read the surrounding module, tests, and docs
- keep `app/main.py` thin
- preserve the FastAPI + browser-native ES module architecture
- prefer deterministic validation over hidden heuristics
- add or update focused tests for behavior changes
- update docs when public behavior or host expectations change

Repo-local guidance for coding agents lives in:

- `AGENTS.md`
- `CLAUDE.md`

## License

[GPL-3.0-or-later](LICENSE)
