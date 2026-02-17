[![CI](https://github.com/pipe-works/pipeworks_axis_descriptor_lab/actions/workflows/ci.yml/badge.svg)](https://github.com/pipe-works/pipeworks_axis_descriptor_lab/actions/workflows/ci.yml) [![Documentation](https://readthedocs.org/projects/pipeworks-axis-descriptor-lab/badge/?version=latest)](https://pipeworks-axis-descriptor-lab.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/pipe-works/pipeworks_axis_descriptor_lab/branch/main/graph/badge.svg)](https://codecov.io/gh/pipe-works/pipeworks_axis_descriptor_lab) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# Axis Descriptor Lab

Tiny web tool for testing how small LLMs (via Ollama) produce _non-authoritative_ descriptive text from a deterministic axis payload. Part of the Pipe-Works project.

## Quick start

```bash
# 1. Install dependencies into the padl virtualenv
pip install -e .

# 2. Copy .env.example and adjust if needed
cp .env.example .env

# 3. Make sure Ollama is running and the model is pulled
ollama pull gemma2:2b

# 4. Start the server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8242
```

Then open **<http://127.0.0.1:8242>** in your browser.

## Usage

1. Choose an example from the dropdown (or paste your own JSON into the textarea).
2. Adjust axis scores with the sliders; labels update automatically in the textarea.
3. Optionally toggle **Auto (policy)** to let the server compute labels from score thresholds, then click **Recompute**.
4. Choose your Ollama model, temperature, and token budget.
5. Click **▶ Generate** to produce a descriptive paragraph.
6. Click **Set as A** to store the output as a baseline, then tweak axes and generate again to see the **Δ Changes** diff.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | SPA shell |
| GET | `/api/examples` | List example names |
| GET | `/api/examples/{name}` | Fetch a named example payload |
| GET | `/api/models` | List locally-pulled Ollama models |
| POST | `/api/generate` | Generate descriptive text |
| POST | `/api/log` | Persist a run log entry |
| POST | `/api/relabel` | Recompute labels from policy |

Interactive API docs: **<http://127.0.0.1:8000/docs>**

## Project layout

```text
axis_descriptor_lab/
├─ README.md
├─ pyproject.toml
├─ .env.example
├─ app/
│  ├─ main.py                # FastAPI app + all routes
│  ├─ ollama_client.py       # HTTP wrapper around Ollama /api/generate
│  ├─ schema.py              # Pydantic v2 models
│  ├─ prompts/
│  │  └─ system_prompt_v01.txt
│  ├─ examples/
│  │  ├─ example_a.json
│  │  └─ example_b.json
│  ├─ static/
│  │  ├─ styles.css
│  │  └─ app.js
│  └─ templates/
│     └─ index.html
└─ logs/
   └─ run_log.jsonl          # created automatically on first log call
```
