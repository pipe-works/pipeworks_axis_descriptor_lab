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

- **`app/main.py`** — FastAPI app bootstrap: creates the application, mounts static/templates, keeps simple endpoints/import routes, and mounts the split route modules.
- **`app/config.py`** — Shared runtime config: app paths, data/log directories, default model, and app version metadata.
- **`app/routes_mud.py`** — `/api/mud/*` proxy router for mud-server authentication and world metadata.
- **`app/routes_save.py`** — Save/export/system-prompt routes.
- **`app/routes_chat.py`** — Chat translation and chat-save routes.
- **`app/services/chat_translation.py`** — Standalone and server-backed chat translation orchestration.
- **`app/services/save_service.py`** — Save/export orchestration helpers.
- **`app/schema/`** — Pydantic v2 models split by domain and re-exported through `app.schema` for backward compatibility.
- **`app/hashing.py`** — IPC normalisation and hash utilities (payload, system prompt, output, composite IPC ID, typed `payload_hash` convenience wrapper).
- **`app/signal_isolation.py`** — NLP pipeline for the Signal Isolation Layer: tokenise (NLTK), lemmatise (WordNet), filter stopwords, compute content-word set delta between two texts. Requires NLTK data packages (punkt_tab, stopwords, wordnet) which are auto-downloaded on first run.
- **`app/chat_renderer.py`** — Unified synchronous Ollama HTTP client. Wraps both `/api/generate` (main page) and `/api/chat` (chat translation page) using httpx. Replaces the old `ollama_client.py`. 10s connect / 120s read timeout.
- **`app/output_validator.py`** — OOC→IC output validator ported from `mud_server`. 7-step pipeline: length check, bracket/OOC leak detection, first-person enforcement, etc. Used by the chat translation endpoint.
- **`app/relabel_policy.py`** — Policy data table (`RELABEL_POLICY`) and `apply_relabel_policy()` function for score-to-label mapping across 11 axes.
- **`app/transformation_map.py`** — Clause-Level Alignment Layer: sentence-aware alignment + token-level diffing to extract clause-level replacement pairs between two texts.
- **`app/save_formatting.py`** — Pure formatting functions for the save system: `save_folder_name()`, `build_output_md()`, `build_baseline_md()`, `build_system_prompt_md()`. No I/O, no app dependencies.
- **`app/save_package.py`** — Save package I/O: writes timestamped session folders under `data/`, builds zip archives, validates manifest checksums on import.
- **`app/file_loaders.py`** — File-loading utilities: `load_default_prompt()`, `load_example()`, `load_prompt()`, `list_example_names()`, `list_prompt_names()`. Reads from `app/prompts/` and `app/examples/`.
- **`app/micro_indicators.py`** — Structural Learning Layer: 10 deterministic heuristic classifiers (`compression`, `expansion`, `embodiment shift`, `abstraction ↑`, `intensity ↑/↓`, `consolidation`, `fragmentation`, `modality shift`, `tone reframing`, `lexical pivot`) that label transformation-map rows. Uses NLTK for POS tagging/sentence segmentation and JSON lexicon data from `app/data/`. Configurable via `IndicatorConfig`.
- **`app/data/`** — JSON lexicon files for micro-indicators: `embodiment_v0_1.json`, `abstraction_v0_1.json`, `intensity_v0_1.json`.

### Frontend (Vanilla JS — ES Modules)

The frontend is split into 21 browser-native ES modules (`app/static/mod-*.js`). No bundler — `<script type="module">` loads the entry point and the browser resolves all imports.

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
- **`mod-navigation.js`** — Page switching between Character Description and Chat Translation (standalone).
- **`mod-chat-state.js`** — Chat Translation page state singleton plus `charDom()` DOM bundle helper.
- **`mod-chat-server-mode.js`** — Mud-server authentication, world selection, active-axis indicators, and server prompt management.
- **`mod-chat-sliders.js`** — Chat Translation slider-panel construction and JSON sync helpers.
- **`mod-chat-game-log.js`** — IPC meta rendering, game-log rendering/copy/export, and chat save packaging.
- **`mod-chat-import.js`** — Chat save import and browser-side state restoration.
- **`mod-chat-translation.js`** — Chat Translation page controller and event wiring.
- **`mod-indicator-modal.js`** — Indicator tooltip text + click-to-open modal with definitions, heuristics, examples, and docs link (standalone, no imports).
- **`mod-tooltip.js`** — JS-positioned tooltip system (standalone, no imports).
- **`mod-theme.js`** — Dark/light theme toggle with localStorage (standalone).

### CSS Architecture (Three-Layer System)

The frontend loads three CSS files in order. No bundler — plain `<link>` tags.

1. **`app/static/pipe-works-fonts.css`** — Self-hosted `@font-face` declarations for 6 OFL font families (16 woff2 files in `app/static/fonts/`). Cacheable, rarely changes.
2. **`app/static/pipe-works-base.css`** — Shared Pipe-Works design system: design tokens (`--col-*`, `--font-*`, `--sp-*`, `--radius-*`), reset, and common components (`.btn`, `.input`, `.select`, `.code-editor`, `.badge`, `.panel`, `.card`, `.modal`, `.divider`, `.spinner`, `.tooltip-bubble`, `.output-box`, scrollbar, utilities). Dark theme is default; light theme activated by `data-theme="light"` on `<html>`. Canonical source: `pipe-works/styles/app/`.
3. **`app/static/styles.css`** — App-specific styles only: four-column grid layout (`panel--left` / `panel--axes` / `panel--centre` / `panel--right`), axis sliders, settings grid, output meta table, diff view, signal isolation, transformation map, collapsible sections, chat translation layout, theme/tooltip toggles, indicator modal. Overrides `.app-header` and `.status-bar` with `position: fixed` (base uses `flex-shrink: 0`).

**Token conventions**: All colours use `--col-*` prefix. Use `color-mix(in srgb, var(--col-token) N%, transparent)` for semi-transparent variants — never hardcoded `rgba()` with literal colours, as these break theme switching.

Supporting files:

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

### Version impact

This project uses `bump-minor-pre-major: true` (`release-please-config.json`), so while the version is pre-1.0 the rules are:

| Type | Version bump | When to use |
|------|-------------|-------------|
| `feat:` | **minor** (0.x → 0.x+1) | Genuinely new user-facing capability: new page, new endpoint, new interactive feature |
| `fix:` | patch (0.0.x → 0.0.x+1) | Bug fixes, including visual/rendering bugs |
| `refactor:` | none | Internal restructuring, layout reorganisation, moving code between files |
| `chore:`, `ci:`, `test:`, `build:`, `docs:`, `style:` | none | Everything else |

### Choosing the right type

The most common mistake is using `feat:` for UI layout or styling work, which triggers an unintended minor bump. Use this guide:

- **New page or major section** → `feat:`
- **New collapsible / toggle / interactive control** → `feat:`
- **CSS layout restructure / column changes** → `refactor:`
- **Visual bug fix (e.g. slider not rendering)** → `fix:`
- **Moving HTML elements around** → `refactor:`
- **Colour, spacing, typography tweaks** → `style:` (or `fix:` if correcting a bug)

When a single PR contains both a `feat:` and a `fix:`, the `feat:` wins — split the PR if the version bump would be misleading.

## Code Style

- Python 3.12+, line length 100 (black + ruff)
- Type hints on Pydantic models and public APIs
- GPL-3.0 license
- Pre-commit hooks: `pre-commit install` then hooks run automatically on commit
- Determinism rules from the broader Pipe-Works ecosystem apply: use `random.Random(seed)`, never global `random.seed()`
