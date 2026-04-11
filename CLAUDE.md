# CLAUDE.md

This file gives repo-local guidance for work in `pipeworks_axis_descriptor_lab`.

## What This Repo Is

Axis Descriptor Lab is a FastAPI application with a browser-native JavaScript
frontend for testing and inspecting non-authoritative LLM output around
deterministic PipeWorks inputs.

Core rule:

- deterministic inputs, hashes, policy logic, and mud-server-owned canonical
  API state are authoritative
- LLM output is ornamental and must never become truth

## Current Luminal Direction

This repo is no longer just a workstation-local toy. It exists inside the real
PipeWorks workspace on `luminal.local`:

- `/srv/work/pipeworks/repos/pipeworks_axis_descriptor_lab`

Its host posture is currently being defined in:

- `/home/aapark/dotfiles/docs/moc/luminal_pipeworks_axis_descriptor_lab_host_preparation.md`

Until that MOC settles more details, work from these assumptions:

- the repo should remain venv-backed for active development
- the repo may also become a deliberate host-managed browser surface
- old workstation-era mirror-file behavior is not a compatibility target
- when canonical truth is in question, prefer `pipeworks_mud_server` APIs over
  duplicated local copies

## Commands

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[docs]"

uvicorn app.main:app --reload --host 127.0.0.1 --port 8242

pytest
pytest -v --cov --cov-report=term

ruff check app tests
black app tests

make -C docs html
```

## Practical Rules

- Read the surrounding code, tests, and docs before editing.
- Keep `app/main.py` thin.
- Put business logic in dedicated Python modules.
- Preserve the current architecture:
  FastAPI backend, browser-native ES modules, no bundler, no database.
- Prefer deterministic validation and transformation logic over hidden client
  heuristics.
- Do not move authoritative logic from Python into frontend code.
- Do not quietly weaken provenance hashing, save/import integrity, or policy
  validation.

## Canonical-Source Boundary

Use the following mental model:

1. `pipeworks_mud_server` owns canonical runtime policy, prompt, and world
   state.
2. Axis Lab may inspect, proxy, compare, compile, or experiment around that
   state.
3. Local lab-only assets are acceptable only when their role is explicit.
4. Legacy mirrored local copies should not survive by inertia.

That means:

- do not add new duplicated local policy or prompt copies unless there is a
  clear documented reason
- do not preserve broad fallback resolution behavior just because it existed
  during the earlier multi-machine development era
- if a local copy remains, label it clearly as canonical, lab-only, or
  transitional

## Host-Shape Caution

Be careful with changes that affect:

- venv layout
- repo-local versus host-managed mutable state
- runtime/log/config paths
- hostname choices
- nginx or `systemd` assumptions

Those decisions should align with the Luminal MOC and host docs, not be
invented ad hoc inside the repo.

## Testing And Docs

Behavior changes need tests. Keep coverage focused near the edited module or
route.

If you change:

- mud-server proxy behavior
- policy or prompt resolution
- save/import behavior
- provenance hashing
- service or host-environment expectations

then update the relevant docs as part of the same work, or explain why not.

## Final Handoff

Report:

- what changed
- which tests ran
- any remaining risks, open host-shape questions, or untested areas
