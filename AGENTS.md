# AGENTS.md

## Foundation Must-Dos

Read and apply these before repo-specific instructions:

- `../.github/.github/docs/AGENT_FOUNDATION.md`
- `../.github/.github/docs/TEST_TAGGING_AND_GITHUB_CHECKLIST.md`

If those foundations conflict with anything here, follow the org-wide
foundation docs first.

## Purpose

This repository is the **Axis Descriptor Lab**: a FastAPI + browser-native JS
tool for inspecting and testing non-authoritative LLM output around
deterministic PipeWorks inputs.

The governing rule is simple:

- deterministic payloads, hashes, server-side rules, and mud-server-owned API
  state are authoritative
- LLM output is ornamental and must never become the source of truth

## Current Luminal Posture

This repo now lives in the real PipeWorks workspace on `luminal.local`:

- workspace root: `/srv/work/pipeworks`
- repo path: `/srv/work/pipeworks/repos/pipeworks_axis_descriptor_lab`

Its Luminal role is under active Management of Change review in:

- `/home/aapark/dotfiles/docs/moc/luminal_pipeworks_axis_descriptor_lab_host_preparation.md`

Treat that MOC as the current source for host-classification direction.

At this stage:

- the repo is already more than clone-only
- the repo is expected to remain venv-backed for active development work
- the repo is also being considered as a deliberate host-managed browser
  surface on Luminal
- legacy workstation-era mirror-file behavior is not a compatibility goal

## Authority Boundary

When deciding where truth lives, use this order:

1. `pipeworks_mud_server` APIs and mud-server-owned canonical state
2. deterministic local validation and formatting logic that exists only to
   inspect or package that canonical state
3. local lab-only experimental assets that are explicitly marked as such
4. never treat LLM output as authoritative

Practical consequences:

- do not expand local mirrored policy or prompt copies just because they are
  convenient
- do not preserve fallback file-resolution layers unless they still serve a
  deliberate supported workflow
- prefer removing stale local duplication over normalizing it
- if a local asset remains, document why it exists and whether it is canonical,
  lab-only, or transitional

## Working Rules

- Read the surrounding module, tests, and docs before editing.
- Keep `app/main.py` thin. Put business logic in dedicated modules.
- Preserve the current architecture:
  FastAPI backend, browser-native ES modules, no bundler, no database.
- Prefer explicit validation and deterministic transformations over hidden
  heuristics.
- Do not move authoritative logic from Python into frontend code.
- Do not quietly weaken provenance hashing, save/import integrity, or policy
  validation behavior.
- When touching mud-server integration, keep the distinction clear between
  lab inspection behavior and canonical server behavior.

## Host And Path Expectations

The older repo-local development story is not the full Luminal story anymore.

Be cautious with changes that assume:

- repo-local `data/` and `logs/` are the right steady-state location for
  host-managed mutable state
- local mirrored files should remain the default source for policy or prompt
  truth
- a Uvicorn entrypoint automatically means the repo should self-define its
  service topology

If work affects venv layout, runtime directories, nginx, `systemd`, host env
files, or hostname choices, align it with the Luminal MOC and host docs rather
than inventing a repo-local pattern.

## Required Commands

- install runtime deps: `pip install -e .`
- install dev deps: `pip install -e ".[dev]"`
- run server: `uvicorn app.main:app --reload --host 127.0.0.1 --port 8242`
- run tests: `pytest`
- run coverage: `pytest -v --cov --cov-report=term`
- lint: `ruff check app tests`
- format: `black app tests`
- build docs: `make -C docs html`

Run the smallest relevant test subset while iterating, then run the broader
affected suite before finishing.

## Testing And Documentation

Behavior changes need tests.

- route changes should verify status codes, payloads, and failure paths
- deterministic logic changes should cover both happy paths and edge cases
- mud-server proxy or canonical-source changes need focused regression coverage
- if behavior changes without tests, explain the gap explicitly

Documentation should stay honest about the current host posture.

- do not describe local mirrored assets as canonical when they are not
- do not document old workstation-era compatibility as if it were still a
  target requirement
- update docs when public behavior, supported workflows, or host expectations
  change

## Release Tags

This repo uses release-please and conventional commits.

Use:

- `feat` for real user-facing capability
- `fix` for bug fixes
- `refactor` for structural changes without new behavior
- `docs`, `style`, `test`, `build`, `ci`, `chore` for their normal narrow use

Do not use `feat` for routine doc cleanup, styling adjustments, or internal
reorganization.

## Handoff

Final handoff should state:

- what changed
- which tests were run
- any remaining risks, open host-classification questions, or untested areas
