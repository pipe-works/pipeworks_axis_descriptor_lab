# AGENTS.md

## Purpose

This repository is the **Axis Descriptor Lab**, a FastAPI + vanilla JavaScript web tool for testing how small LLMs produce **non-authoritative** descriptive text from deterministic axis payloads in the Pipe-Works ecosystem.

The core rule of the system is:

- The deterministic payload, policy, hashes, and server-side rules are authoritative.
- The LLM is ornamental and must never become the source of truth.

Agents working in this repository must preserve that distinction in code, tests, docs, and review decisions.

## Repository Shape

- `app/main.py`: thin FastAPI routing layer and application bootstrap.
- `app/schema.py`: Pydantic v2 request/response models with OpenAPI-facing field descriptions.
- `app/chat_renderer.py`: synchronous Ollama HTTP client.
- `app/signal_isolation.py`, `app/transformation_map.py`, `app/micro_indicators.py`: deterministic text-analysis layers.
- `app/relabel_policy.py`: score-to-label policy mapping.
- `app/save_formatting.py`, `app/save_package.py`: save/export/import helpers.
- `app/file_loaders.py`: prompt/example loading helpers.
- `app/static/`: browser-native ES modules and CSS. No bundler.
- `app/templates/index.html`: single-page shell rendered by Jinja2.
- `tests/`: pytest suite covering routes, domain modules, persistence, and frontend-adjacent backend behavior.
- `docs/`: Sphinx documentation and narrative guides.

## Working Rules

- Read the surrounding module, tests, and docs before editing. Match local patterns instead of introducing a new style.
- Keep `app/main.py` thin. Put business logic in dedicated modules.
- Prefer deterministic logic and explicit validation over heuristics hidden in the frontend.
- Preserve the current architecture: FastAPI backend, vanilla JS frontend, no build step, no database.
- Do not quietly weaken validation, provenance hashing, or reproducibility behavior.
- Do not replace server-owned policy logic with client-side shortcuts.

## Required Commands

- Install runtime deps: `pip install -e .`
- Install dev deps: `pip install -e ".[dev]"`
- Run server: `uvicorn app.main:app --reload --host 127.0.0.1 --port 8242`
- Run tests: `pytest`
- Run coverage: `pytest -v --cov --cov-report=term`
- Lint: `ruff check app tests`
- Format: `black app tests`
- Build docs: `make -C docs html`

Run the smallest relevant test subset during iteration, then run the broader affected suite before finishing.

## CI Fast Lane

- CI uses the shared pipe-works reusable workflow with stable required checks (`All Checks Passed`, `Secret Scan (Gitleaks)`).
- Content-only pull requests can take the fast path: `Change Classification` + `Content Validation` instead of the full Python matrix.
- Current content-path scope in this repo: `app/worlds/**`, `app/lab_only/**`, `docs/**`, and `*.md`.

## Documentation Standard

Detailed documentation is mandatory in this repository.

- Every Python module must have a top-level module docstring explaining purpose, boundaries, and important design decisions.
- Every public Python class, function, method, and fixture must have a detailed docstring.
- Pydantic fields should keep meaningful `description=` metadata so FastAPI docs stay useful.
- Every JavaScript module should begin with a detailed header comment describing ownership, data flow, and major responsibilities.
- Exported JavaScript functions should use JSDoc-style comments when the behavior is not trivially obvious.
- Add inline comments for non-obvious logic, invariants, protocol details, and edge-case handling.
- Do not add noise comments that restate syntax. Comments must explain intent, constraints, or reasoning.

## Testing Standard

Tests are mandatory for behavior changes.

- Every bug fix needs a regression test.
- Every new endpoint, branch, validation rule, or persistence behavior needs direct test coverage.
- For Python backend changes, prefer pytest coverage close to the edited module.
- For route changes, verify HTTP status, response body, and error handling paths.
- For deterministic logic, test both happy paths and edge cases.
- When changing hashes, policy mapping, prompt loading, save/import behavior, or mud-server proxy behavior, add or update focused tests.
- If a change is intentionally not tested, explain the gap clearly in the final handoff.

## Python Conventions

- Target Python `3.12+`.
- Keep line length at `100`.
- Use type hints on public APIs and non-trivial internal helpers.
- Keep models and pure helpers small and explicit.
- Prefer pure functions for deterministic transformations where practical.
- Raise explicit errors instead of silently correcting invalid inputs unless the surrounding module already defines a softer contract.
- Keep docstrings, comments, and naming aligned with the repository's formal style.

## Frontend Conventions

- Use browser-native ES modules only. Do not introduce a bundler or framework.
- Keep responsibilities split across the existing `mod-*.js` files instead of growing one large script.
- Preserve the current design system layering:
  - `pipe-works-fonts.css`
  - `pipe-works-base.css`
  - `styles.css`
- Use existing CSS tokens and theme-aware color variables. Avoid hardcoded colors when a project token exists.
- Keep the frontend as an orchestration/UI layer. Do not move authoritative logic from Python into JavaScript.

## GitHub And Release Tags

This repository uses **release-please** and conventional commit tags. Use the correct tag because it affects versioning and changelog output.

Allowed tags in practice:

- `feat`: new user-facing capability. In pre-1.0, this causes a minor bump.
- `fix`: bug fix.
- `perf`: performance improvement.
- `revert`: explicit revert.
- `docs`: documentation-only change.
- `style`: styling or formatting-only change.
- `chore`: maintenance work.
- `refactor`: internal restructuring without a user-facing feature.
- `test`: test-only changes.
- `build`: packaging/build changes.
- `ci`: CI workflow changes.

Tag selection rules:

- Do **not** use `feat` for layout tweaks, CSS cleanup, comment-only work, test-only work, or internal refactors.
- Prefer `refactor` for structural changes that do not add behavior.
- Prefer `style` for presentation-only changes.
- Prefer `docs` for README, Sphinx, or comment/docstring-only updates.
- Split unrelated work so a real `feat` does not hide routine fixes or refactors.

## Change Checklist

Before finishing work, verify the following:

- Architecture remains consistent with the thin-route / domain-module split.
- New or changed code includes detailed comments and docstrings matching local style.
- Relevant tests were added or updated.
- `pytest` was run for affected coverage, or any gap is explicitly reported.
- `ruff` and `black` concerns were addressed for changed Python files.
- Documentation was updated when public behavior, endpoints, prompts, or workflows changed.

## Handoff Expectations

Final handoff should state:

- What changed.
- Which tests were run.
- Any remaining risks, follow-ups, or untested areas.

If a task touches behavior without adding tests or documentation, treat the work as incomplete unless the user explicitly directs otherwise.
