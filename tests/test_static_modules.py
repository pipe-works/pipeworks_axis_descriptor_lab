"""Tests for the ES module structure introduced by the app.js → mod-*.js refactor.

Verifies that:
  1. All module files are served at /static/ with correct content type.
  2. The HTML template references the ES module entry point.
  3. The old monolithic app.js is no longer served.
  4. Each module contains its expected imports and exports.
  5. The import graph has no circular dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ── Module manifest ─────────────────────────────────────────────────────────
# Every mod-*.js file that should exist, with expected exports and imports.

MODULE_MANIFEST: dict[str, dict] = {
    "mod-utils.js": {
        "exports": [
            "clamp",
            "debounce",
            "safeParse",
            "tokenise",
            "lcsWordDiff",
            "extractTransformationRows",
            "cryptoRandomFloat",
            "makePlaceholder",
        ],
        "imports_from": [],  # leaf — no imports
    },
    "mod-state.js": {
        "exports": ["state", "dom"],
        "imports_from": [],  # leaf — no imports
    },
    "mod-axis-policy.js": {
        "exports": [
            "CANONICAL_AXIS_ORDER",
            "AXIS_SCORE_STEP",
            "orderAxisKeys",
            "quantiseAxisScore",
            "formatAxisScore",
        ],
        "imports_from": [],  # leaf — no imports
    },
    "mod-status.js": {
        "exports": ["setStatus"],
        "imports_from": ["mod-state.js"],
    },
    "mod-sync.js": {
        "exports": [
            "syncJsonTextarea",
            "setJsonBadge",
            "updateSystemPromptBadge",
            "buildSlidersFromJson",
            "getModelName",
            "getOllamaHost",
            "resolveSeed",
            "refreshModels",
            "wireSyncEvents",
        ],
        "imports_from": ["mod-state.js", "mod-axis-policy.js", "mod-utils.js", "mod-status.js"],
    },
    "mod-source-paths.js": {
        "exports": ["resolveArtifactSourcePath", "renderSourceHint"],
        "imports_from": [],  # shared leaf helper
    },
    "mod-pipeline-build-state.js": {
        "exports": [
            "PIPELINE_STAGE_ORDER",
            "PIPELINE_STAGE_STATUS",
            "pipelineBuildState",
            "resetPipelineBuildState",
        ],
        "imports_from": [],  # pipeline scaffold state leaf
    },
    "mod-pipeline-build-api.js": {
        "exports": [
            "fetchMudSession",
            "fetchMudWorlds",
            "selectMudWorld",
            "fetchMudWorldConfig",
            "fetchMudImagePolicyBundle",
            "fetchPipelineBuildBootstrap",
            "resolvePipelineImageSelection",
            "generatePipelineConditionAxis",
            "fetchLocalAxisPayloads",
            "fetchLocalAxisPayload",
            "relabelAxisPayload",
            "compileImagePrompt",
        ],
        "imports_from": [],  # pipeline API helper leaf
    },
    "mod-pipeline-build-hash.js": {
        "exports": ["stableStringify", "sha256Hex", "hashNormalizedPayload"],
        "imports_from": [],  # deterministic hash helper leaf
    },
    "mod-pipeline-build.js": {
        "exports": ["initPipelineBuild", "wirePipelineBuildEvents"],
        "imports_from": [
            "mod-state.js",
            "mod-status.js",
            "mod-pipeline-build-state.js",
            "mod-pipeline-build-api.js",
            "mod-pipeline-build-hash.js",
        ],
    },
    "mod-loaders.js": {
        "exports": [
            "loadExampleList",
            "loadExample",
            "loadPromptList",
            "loadPrompt",
            "wireLoaderEvents",
        ],
        "imports_from": ["mod-state.js", "mod-status.js", "mod-sync.js", "mod-source-paths.js"],
    },
    "mod-generate.js": {
        "exports": ["generate", "wireGenerateEvents"],
        "imports_from": [
            "mod-state.js",
            "mod-utils.js",
            "mod-status.js",
            "mod-sync.js",
            "mod-diff.js",
        ],
    },
    "mod-diff.js": {
        "exports": [
            "updateDiff",
            "updateSignalIsolation",
            "updateTransformationMap",
            "wireDiffEvents",
        ],
        "imports_from": [
            "mod-state.js",
            "mod-utils.js",
            "mod-status.js",
            "mod-indicator-modal.js",
        ],
    },
    "mod-axis-actions.js": {
        "exports": ["relabel", "randomiseAxes", "wireAxisEvents"],
        "imports_from": [
            "mod-state.js",
            "mod-axis-policy.js",
            "mod-utils.js",
            "mod-status.js",
            "mod-sync.js",
        ],
    },
    "mod-persistence.js": {
        "exports": [
            "saveRun",
            "exportSave",
            "importSave",
            "restoreSessionState",
            "logRun",
            "wirePersistenceEvents",
        ],
        "imports_from": [
            "mod-state.js",
            "mod-utils.js",
            "mod-status.js",
            "mod-sync.js",
            "mod-diff.js",
        ],
    },
    "mod-indicator-modal.js": {
        "exports": [
            "getIndicatorTooltip",
            "openIndicatorModal",
            "wireIndicatorModalEvents",
        ],
        "imports_from": [],  # standalone leaf module
    },
    "mod-tooltip.js": {
        "exports": ["wireTooltipToggle"],
        "imports_from": [],  # standalone
    },
    "mod-theme.js": {
        "exports": ["wireThemeToggle"],
        "imports_from": [],  # standalone
    },
    "mod-navigation.js": {
        "exports": ["wireNavigationEvents"],
        "imports_from": ["mod-state.js"],
    },
    "mod-artifact-editor.js": {
        "exports": ["initArtifactEditor", "wireArtifactEditorEvents"],
        "imports_from": ["mod-state.js", "mod-status.js", "mod-source-paths.js"],
    },
    "mod-chat-state.js": {
        "exports": ["chatState", "charDom"],
        "imports_from": ["mod-state.js"],
    },
    "mod-chat-server-mode.js": {
        "exports": [
            "isServerMode",
            "updateModeBadge",
            "checkSession",
            "doLogin",
            "doLogout",
            "selectWorld",
            "handleSessionExpired",
            "toggleServerControls",
            "applyActiveAxesIndicators",
            "clearActiveAxesIndicators",
            "getEffectiveSystemPrompt",
            "wireServerModeEvents",
        ],
        "imports_from": ["mod-state.js", "mod-status.js", "mod-chat-state.js"],
    },
    "mod-chat-sliders.js": {
        "exports": ["syncJsonTextarea", "setJsonBadge", "buildChatSliders"],
        "imports_from": [
            "mod-axis-policy.js",
            "mod-utils.js",
            "mod-chat-state.js",
            "mod-chat-server-mode.js",
        ],
    },
    "mod-chat-game-log.js": {
        "exports": [
            "buildIpcMetaTable",
            "renderTranslationResult",
            "appendGameEntry",
            "copyGameLogTxt",
            "copyGameLogMd",
            "saveChatLog",
            "wireGameLogEvents",
        ],
        "imports_from": [
            "mod-state.js",
            "mod-status.js",
            "mod-chat-state.js",
            "mod-chat-server-mode.js",
        ],
    },
    "mod-chat-import.js": {
        "exports": ["restoreChatSessionState", "importChatSave", "wireChatImportEvents"],
        "imports_from": [
            "mod-state.js",
            "mod-status.js",
            "mod-chat-state.js",
            "mod-chat-sliders.js",
            "mod-chat-game-log.js",
        ],
    },
    "mod-chat-translation.js": {
        "exports": ["translate", "initChatTranslation", "wireChatTranslationEvents"],
        "imports_from": [
            "mod-state.js",
            "mod-axis-policy.js",
            "mod-utils.js",
            "mod-status.js",
            "mod-chat-state.js",
            "mod-chat-sliders.js",
            "mod-chat-server-mode.js",
            "mod-chat-game-log.js",
            "mod-chat-import.js",
            "mod-source-paths.js",
        ],
    },
    "mod-events.js": {
        "exports": ["wireEvents"],
        "imports_from": [
            "mod-sync.js",
            "mod-loaders.js",
            "mod-generate.js",
            "mod-diff.js",
            "mod-axis-actions.js",
            "mod-persistence.js",
            "mod-indicator-modal.js",
            "mod-navigation.js",
            "mod-chat-translation.js",
            "mod-artifact-editor.js",
            "mod-pipeline-build.js",
        ],
    },
    "mod-init.js": {
        "exports": [],  # side-effect only (DOMContentLoaded listener)
        "imports_from": [
            "mod-state.js",
            "mod-status.js",
            "mod-theme.js",
            "mod-tooltip.js",
            "mod-events.js",
            "mod-loaders.js",
            "mod-chat-translation.js",
            "mod-artifact-editor.js",
            "mod-pipeline-build.js",
        ],
    },
}

ALL_MODULE_NAMES = sorted(MODULE_MANIFEST.keys())


# ── Helpers ─────────────────────────────────────────────────────────────────


def _read_module(name: str) -> str:
    """Read a module file from the static directory."""
    path = Path(__file__).resolve().parent.parent / "app" / "static" / name
    return path.read_text(encoding="utf-8")


def _read_styles() -> str:
    """Read the app stylesheet."""
    path = Path(__file__).resolve().parent.parent / "app" / "static" / "styles.css"
    return path.read_text(encoding="utf-8")


def _read_template() -> str:
    """Read the main index template."""
    path = Path(__file__).resolve().parent.parent / "app" / "templates" / "index.html"
    return path.read_text(encoding="utf-8")


# ── 1. Static file serving ──────────────────────────────────────────────────


class TestModuleServing:
    """All 21 mod-*.js files are served via /static/ with correct content type."""

    @pytest.mark.parametrize("module_name", ALL_MODULE_NAMES)
    def test_module_served_with_200(self, client: TestClient, module_name: str) -> None:
        """Each module file returns HTTP 200."""
        res = client.get(f"/static/{module_name}")
        assert res.status_code == 200, f"/static/{module_name} returned {res.status_code}"

    @pytest.mark.parametrize("module_name", ALL_MODULE_NAMES)
    def test_module_content_type_is_javascript(self, client: TestClient, module_name: str) -> None:
        """Each module file is served with a JavaScript content type."""
        res = client.get(f"/static/{module_name}")
        ct = res.headers.get("content-type", "")
        assert (
            "javascript" in ct
        ), f"/static/{module_name} content-type is '{ct}', expected JavaScript"

    @pytest.mark.parametrize("module_name", ALL_MODULE_NAMES)
    def test_module_is_not_empty(self, client: TestClient, module_name: str) -> None:
        """Each module file has non-trivial content."""
        res = client.get(f"/static/{module_name}")
        assert len(res.text) > 50, f"/static/{module_name} is suspiciously small"


# ── 2. HTML template references ─────────────────────────────────────────────


class TestHtmlTemplate:
    """The HTML template correctly references the ES module entry point."""

    def test_entry_point_is_mod_init(self, client: TestClient) -> None:
        """index.html contains a <script type="module" src="/static/mod-init.js">."""
        res = client.get("/")
        assert res.status_code == 200
        assert 'type="module"' in res.text
        assert 'src="/static/mod-init.js"' in res.text

    def test_old_app_js_not_referenced(self, client: TestClient) -> None:
        """index.html does not reference the old monolithic app.js."""
        res = client.get("/")
        assert 'src="/static/app.js"' not in res.text


# ── 3. Old app.js removed ──────────────────────────────────────────────────


class TestOldAppJsRemoved:
    """The old monolithic app.js file has been deleted."""

    def test_app_js_not_on_disk(self) -> None:
        """app/static/app.js should not exist on disk."""
        path = Path(__file__).resolve().parent.parent / "app" / "static" / "app.js"
        assert not path.exists(), "app.js should have been deleted after the refactor"

    def test_app_js_returns_404(self, client: TestClient) -> None:
        """/static/app.js should return 404."""
        res = client.get("/static/app.js")
        assert res.status_code == 404


# ── 4. Module exports ──────────────────────────────────────────────────────


class TestModuleExports:
    """Each module file contains its expected export declarations."""

    @pytest.mark.parametrize(
        "module_name,expected_exports",
        [
            (name, info["exports"])
            for name, info in MODULE_MANIFEST.items()
            if info["exports"]  # skip mod-init.js (no exports)
        ],
        ids=lambda val: val if isinstance(val, str) else None,
    )
    def test_expected_exports_present(self, module_name: str, expected_exports: list[str]) -> None:
        """Each declared export name appears in an export statement."""
        content = _read_module(module_name)
        for export_name in expected_exports:
            # Match: export function foo, export async function foo,
            #        export const foo, export let foo
            pattern = rf"export\s+(?:async\s+)?(?:function|const|let)\s+{re.escape(export_name)}\b"
            assert re.search(
                pattern, content
            ), f"Expected export '{export_name}' not found in {module_name}"


# ── 5. Module imports ──────────────────────────────────────────────────────


class TestModuleImports:
    """Each module file imports from the expected set of modules."""

    @pytest.mark.parametrize(
        "module_name,expected_imports",
        [
            (name, info["imports_from"])
            for name, info in MODULE_MANIFEST.items()
            if info["imports_from"]
        ],
        ids=lambda val: val if isinstance(val, str) else None,
    )
    def test_expected_imports_present(self, module_name: str, expected_imports: list[str]) -> None:
        """Each expected import source appears in an import statement."""
        content = _read_module(module_name)
        for dep in expected_imports:
            # Match: import ... from "./mod-foo.js"
            assert (
                f'"./{dep}"' in content
            ), f"Expected import from '{dep}' not found in {module_name}"

    @pytest.mark.parametrize("module_name", ALL_MODULE_NAMES)
    def test_no_unexpected_self_import(self, module_name: str) -> None:
        """No module imports from itself."""
        content = _read_module(module_name)
        assert f'"./{module_name}"' not in content, f"{module_name} imports from itself"


class TestPipelineBuildContracts:
    """Lightweight source contracts for Pipeline Build behavior."""

    def test_unauthenticated_flow_preserves_entered_state_by_default(self) -> None:
        """401 handling should preserve entered state unless explicitly disabled."""
        content = _read_module("mod-pipeline-build.js")
        assert (
            "function applyUnauthenticatedState(errorMessage = null, { preserveEnteredState = true } = {})"
            in content
        )
        assert "if (!preserveEnteredState)" in content
        assert "Entered state preserved for re-auth." in content

    def test_action_log_messages_are_sanitised(self) -> None:
        """Action-log messages should redact common secret-bearing fields."""
        content = _read_module("mod-pipeline-build.js")
        assert "function sanitiseActionLogMessage(rawMessage)" in content
        assert "password|token|secret|authorization" in content
        assert "message: safeMessage" in content

    def test_axis_source_hint_is_rendered_for_canonical_stage4_generation(self) -> None:
        """Stage 4 should surface canonical condition-axis API source hints."""
        content = _read_module("mod-pipeline-build.js")
        assert "dom.pipelineAxisSourceHint" in content
        assert "Condition Axis API: /api/mud/pipeline-build/generate-condition-axis" in content

    def test_generated_axis_payload_schema_validation_path_exists(self) -> None:
        """Generated condition-axis payloads should be schema-validated before stage advance."""
        content = _read_module("mod-pipeline-build.js")
        assert "function validateAxisPayloadSchema(payload)" in content
        assert "Canonical condition axis payload schema error" in content
        assert "payload.axes" in content
        assert "label must be a non-empty string" in content

    def test_canonical_condition_axis_generation_is_wired(self) -> None:
        """Stage 4 should call canonical generation endpoint with seed controls."""
        content = _read_module("mod-pipeline-build.js")
        api_content = _read_module("mod-pipeline-build-api.js")

        assert "function buildConditionAxisGenerationRequest()" in content
        assert 'pipelineBuildState.axis.seedMode === "fixed"' in content
        assert "async function handleGenerateConditionAxis()" in content
        assert "const payload = await generatePipelineConditionAxis(requestBody);" in content
        assert 'dom.pipelineAxisGenerate?.addEventListener("click", () => {' in content

        assert "export async function generatePipelineConditionAxis(body)" in api_content
        assert '"/api/mud/pipeline-build/generate-condition-axis"' in api_content

    def test_condition_axis_unsupported_error_surfaces_structured_detail(self) -> None:
        """Stage 4 unsupported-path errors should surface structured upstream detail text."""
        content = _read_module("mod-pipeline-build.js")
        assert 'err.code === "PIPELINE_UPSTREAM_UNSUPPORTED"' in content
        assert "Pipeline Build — condition axis generation unsupported: ${detail}" in content

    def test_stage_progression_can_mark_downstream_stages_complete(self) -> None:
        """Resolve/compile results should promote downstream stage statuses."""
        content = _read_module("mod-pipeline-build.js")
        assert "const hasCompileResult = Boolean(pipelineBuildState.compile.result);" in content
        assert "const hasResolvePreview = Boolean(pipelineBuildState.resolve.result);" in content
        assert "const previewStatus = hasCompileResult || hasResolvePreview" in content
        assert 'setStageStatus("block_selection", previewStatus);' in content
        assert 'setStageStatus("descriptor_tone", previewStatus);' in content
        assert 'setStageStatus("composition_hashes", previewStatus);' in content
        assert 'setStageStatus("compile_output",' in content

    def test_composition_preview_exposes_hash_contract(self) -> None:
        """Composition preview must include policy/axis/compiler hash fields."""
        content = _read_module("mod-pipeline-build.js")
        assert "function renderCompositionPreview()" in content
        assert '`policy_hash: ${pipelineBuildState.policyHash || "(not loaded)"}`' in content
        assert '`axis_hash: ${pipelineBuildState.axisHash || "(not computed)"}`' in content
        assert (
            '`compiler_input_hash: ${pipelineBuildState.compilerInputHash || "(not computed)"}`'
            in content
        )
        assert '"resolve_preview"' in content
        assert '"policy_bundle/runtime"' in content

    def test_hash_input_disclosure_excludes_compiled_prompt(self) -> None:
        """Hash contract must explicitly exclude final compiled prompt text."""
        content = _read_module("mod-pipeline-build.js")
        assert "AXIS_HASH_INPUT_FIELDS" in content
        assert "COMPILER_INPUT_HASH_FIELDS" in content
        assert '"excluded_from_hashes: compiled_prompt"' in content

    def test_hash_helpers_and_recompute_contract_exist(self) -> None:
        """Hash helper module should define deterministic normalization + hashing."""
        hash_content = _read_module("mod-pipeline-build-hash.js")
        pipeline_content = _read_module("mod-pipeline-build.js")

        assert "function stableStringify(value)" in hash_content
        assert "Object.keys(node).sort()" in hash_content
        assert 'crypto.subtle.digest("SHA-256", data)' in hash_content
        assert "return sha256Hex(stableStringify(value));" in hash_content

        assert "async function recomputeHashes()" in pipeline_content
        assert "pipelineBuildState.axisHash" in pipeline_content
        assert "pipelineBuildState.compilerInputHash" in pipeline_content
        assert "await hashNormalizedPayload(axisPayload)" in pipeline_content
        assert "await hashNormalizedPayload(compileRequest)" in pipeline_content

    def test_request_body_builders_have_gating_and_expected_fields(self) -> None:
        """Resolve/compile request builders should gate by stage validity and include expected fields."""
        content = _read_module("mod-pipeline-build.js")

        assert "function buildCompileRequest()" in content
        assert "function buildResolveRequest()" in content
        assert "if (!pipelineBuildState.selectedWorldId) return null;" in content
        assert "if (!isIdentityValid()) return null;" in content
        assert "if (!isAxisPayloadValid(pipelineBuildState.axis.payload)) return null;" in content
        assert "if (!worldIdsMatch()) return null;" in content

        assert "world_id: pipelineBuildState.selectedWorldId" in content
        assert "species: pipelineBuildState.identity.species" in content
        assert "gender: pipelineBuildState.identity.gender" in content
        assert "axes: pipelineBuildState.axis.payload.axes" in content
        assert "world_context: runtime.worldContext" in content
        assert "occupation_signals: runtime.occupationSignals" in content

    def test_stage_transition_rules_cover_auth_world_policy_identity_axis(self) -> None:
        """Stage progression should explicitly gate downstream stages in canonical order."""
        content = _read_module("mod-pipeline-build.js")

        assert "function applyStageProgression()" in content
        assert "if (!isAuthenticated)" in content
        assert "if (!hasWorld)" in content
        assert "if (policyStatus !== PIPELINE_STAGE_STATUS.COMPLETE)" in content
        assert "if (!identityValid)" in content
        assert "if (axisValid)" in content
        assert "lockAfterAxis();" in content
        assert 'pipelineBuildState.activeStage = "session_world";' in content
        assert 'pipelineBuildState.activeStage = "identity";' in content
        assert 'pipelineBuildState.activeStage = "axis_input";' in content

    def test_session_world_policy_fetch_flow_is_wired(self) -> None:
        """Session/world/policy bootstrap flow should call API helpers in sequence."""
        content = _read_module("mod-pipeline-build.js")

        assert "async function refreshSessionAndWorlds" in content
        assert "const session = await fetchMudSession();" in content
        assert "const worldsPayload = await fetchMudWorlds();" in content
        assert "const preferredWorld = derivePreferredWorld(session, worlds);" in content
        assert "await applyWorldSelection(preferredWorld, { quiet: true });" in content

        assert "async function loadPolicyBundleForWorld" in content
        assert "const bootstrap = await fetchPipelineBuildBootstrap(worldId);" in content
        assert "pipelineBuildState.policyBundle = bundle;" in content
        assert "pipelineBuildState.policySource = bootstrap.policy_source || null;" in content
        assert (
            "pipelineBuildState.worldConfig = bootstrap.world_summary?.world_config || null;"
            in content
        )

    def test_policy_source_display_uses_truthful_source_metadata(self) -> None:
        """Policy source row should use bootstrap policy_source metadata, not inferred local paths."""
        content = _read_module("mod-pipeline-build.js")
        assert "function resolvePolicySourceBadgeInfo(policySource)" in content
        assert "function formatPolicySourceReference(policySource, bundle)" in content
        assert "Policy source: Offline mode (no canonical policy endpoint)." in content
        assert 'if (sourceKind === "mud_server_canonical")' in content
        assert "served_via" in content

    def test_policy_bundle_view_file_modal_is_wired_read_only(self) -> None:
        """Policy Bundle row should expose a read-only file modal trigger + wiring."""
        content = _read_module("mod-pipeline-build.js")
        assert "function setPolicyBundleFileModalOpen(isOpen)" in content
        assert "function buildPolicyBundleFileDocument()" in content
        assert "function renderPolicyBundleFileModal()" in content
        assert "read_only: true" in content
        assert 'dom.pipelinePolicyViewFile?.addEventListener("click", () => {' in content
        assert (
            "dom.pipelinePolicyFileModalContent.textContent = JSON.stringify(payload, null, 2);"
            in content
        )

    def test_pipeline_refreshes_session_world_when_mud_mode_changes(self) -> None:
        """Pipeline Build should refresh Session + World when global mud mode changes."""
        content = _read_module("mod-pipeline-build.js")
        assert 'document.addEventListener("mud-session-context-changed", () => {' in content
        assert 'dom.pagePipelineBuild?.classList.contains("hidden")' in content
        assert "refreshSessionAndWorlds({ quiet: true });" in content

    def test_mode_switch_dispatches_mud_session_context_changed_event(self) -> None:
        """Runtime mode switches should broadcast a context-change event."""
        content = _read_module("mod-chat-server-mode.js")
        assert "function dispatchMudSessionContextChanged(reason)" in content
        assert 'new CustomEvent("mud-session-context-changed"' in content
        assert 'dispatchMudSessionContextChanged("runtime_mode_changed");' in content

    def test_compile_request_contract_flow_is_wired(self) -> None:
        """Compile action should build request payload then post to compile endpoint."""
        content = _read_module("mod-pipeline-build.js")

        assert "async function handleCompileRequest()" in content
        assert "const requestBody = buildCompileRequest();" in content
        assert "const result = await compileImagePrompt(requestBody);" in content
        assert "pipelineBuildState.compile.result = result;" in content
        assert "pipelineBuildState.policyHash = String(result.policy_hash);" in content
        assert "pipelineBuildState.axisHash = String(result.axis_hash);" in content

    def test_401_paths_call_unauthenticated_state_and_relock(self) -> None:
        """Pipeline fetch/resolve/compile flows should handle 401 via re-auth relock helper."""
        content = _read_module("mod-pipeline-build.js")

        assert (
            "function applyUnauthenticatedState(errorMessage = null, { preserveEnteredState = true } = {})"
            in content
        )
        assert 'setStageStatus("policy_bundle", PIPELINE_STAGE_STATUS.LOCKED);' in content
        assert 'setStageStatus("identity", PIPELINE_STAGE_STATUS.LOCKED);' in content
        assert 'setStageStatus("axis_input", PIPELINE_STAGE_STATUS.LOCKED);' in content
        assert "lockAfterAxis();" in content

        # 401 handling exists across world/session/resolve/compile flows.
        assert "if (err instanceof PipelineApiError && err.status === 401)" in content
        assert "applyUnauthenticatedState(detail);" in content
        assert "mud session expired. Please reconnect." in content

    def test_source_hints_present_for_world_policy_axis_and_species_inputs(self) -> None:
        """Pipeline UI should surface source hints for world, policy bundle, axis generation, and species."""
        template = _read_template()
        content = _read_module("mod-pipeline-build.js")

        assert 'id="pipeline-world-source-hint"' in template
        assert 'id="pipeline-policy-source-hint"' in template
        assert 'id="pipeline-axis-source-hint"' in template
        assert 'id="pipeline-species-source-hint"' in template
        assert "dom.pipelineAxisSourceHint" in content
        assert "dom.pipelineSpeciesSourceHint" in content

    def test_pipeline_api_error_parser_supports_code_and_stage_fields(self) -> None:
        """Pipeline API helper should preserve structured error code/stage metadata."""
        content = _read_module("mod-pipeline-build-api.js")
        assert "this.code = code;" in content
        assert "this.stage = stage;" in content
        assert 'if (typeof body.code === "string") code = body.code;' in content
        assert 'if (typeof body.stage === "string") stage = body.stage;' in content

    def test_stage_list_keyboard_navigation_is_wired(self) -> None:
        """Stage list should support arrow navigation and Enter/Space activation."""
        content = _read_module("mod-pipeline-build.js")
        assert "function wireStageListInteractions()" in content
        assert 'event.key === "ArrowDown"' in content
        assert 'event.key === "ArrowUp"' in content
        assert 'event.key === "Enter" || event.key === " "' in content
        assert "focusStageControl(stageKey)" in content


class TestPipelineBuildStyles:
    """CSS contracts for Pipeline Build accessibility/responsive behavior."""

    def test_pipeline_mobile_fallback_layout_present(self) -> None:
        """Pipeline page should define a stacked mobile layout fallback."""
        styles = _read_styles()
        assert "@media (max-width: 1100px)" in styles
        assert ".pipeline-build-grid" in styles
        assert "flex-direction: column" in styles

    def test_pipeline_stage_focus_style_present(self) -> None:
        """Keyboard focus for stage rows should be visually apparent."""
        styles = _read_styles()
        assert ".pipeline-build-grid #pipeline-stage-list li:focus-visible" in styles
        assert "box-shadow" in styles


# ── 6. No circular dependencies ────────────────────────────────────────────


class TestNoCycles:
    """The import graph is acyclic (no circular dependencies)."""

    def test_import_graph_is_dag(self) -> None:
        """Topological sort of the import graph succeeds (no cycles)."""
        # Build adjacency: module → set of modules it imports from
        graph: dict[str, set[str]] = {}
        for name, info in MODULE_MANIFEST.items():
            graph[name] = set(info["imports_from"])

        # Kahn's algorithm for topological sort
        in_degree: dict[str, int] = {name: 0 for name in graph}
        for name, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] = in_degree.get(dep, 0)  # ensure exists
                    # This counts how many modules depend on `dep`, but for
                    # cycle detection we need in-degree of `name`
                    pass

        # Recompute: in_degree[x] = number of modules that x imports from
        # Actually for cycle detection: in_degree[x] = number of modules
        # that import x (reverse edges)
        in_degree = {name: 0 for name in graph}
        for name, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[name] += 1  # name depends on dep

        # Wait — simpler: just do DFS cycle detection
        WHITE, GREY, BLACK = 0, 1, 2
        colour: dict[str, int] = {name: WHITE for name in graph}
        cycle_path: list[str] = []

        def dfs(node: str) -> bool:
            """Return True if a cycle is found."""
            colour[node] = GREY
            for dep in graph[node]:
                if dep not in colour:
                    continue  # external dependency, skip
                if colour[dep] == GREY:
                    cycle_path.append(f"{node} → {dep}")
                    return True
                if colour[dep] == WHITE:
                    if dfs(dep):
                        cycle_path.append(f"{node} → {dep}")
                        return True
            colour[node] = BLACK
            return False

        has_cycle = False
        for node in graph:
            if colour[node] == WHITE:
                if dfs(node):
                    has_cycle = True
                    break

        assert not has_cycle, f"Circular dependency detected: {' | '.join(reversed(cycle_path))}"


# ── 7. Module file-level documentation ─────────────────────────────────────


class TestModuleDocumentation:
    """Each module file has a file-level JSDoc header comment."""

    @pytest.mark.parametrize("module_name", ALL_MODULE_NAMES)
    def test_has_file_header_comment(self, module_name: str) -> None:
        """Each module starts with a JSDoc block comment."""
        content = _read_module(module_name)
        assert content.lstrip().startswith(
            "/**"
        ), f"{module_name} does not start with a JSDoc comment block"

    @pytest.mark.parametrize("module_name", ALL_MODULE_NAMES)
    def test_header_mentions_module_name(self, module_name: str) -> None:
        """The file header mentions the module filename."""
        content = _read_module(module_name)
        # The module name should appear in the first few lines
        header = content[:500]
        assert module_name in header, f"{module_name} header does not mention its own filename"


# ── 8. Module count ────────────────────────────────────────────────────────


class TestModuleCount:
    """The module count on disk should stay in sync with the declared manifest."""

    def test_module_count_matches_manifest(self) -> None:
        """The static directory contains exactly the modules declared in the manifest."""
        static_dir = Path(__file__).resolve().parent.parent / "app" / "static"
        mod_files = sorted(p.name for p in static_dir.glob("mod-*.js"))
        assert len(mod_files) == len(MODULE_MANIFEST), (
            f"Expected {len(MODULE_MANIFEST)} mod-*.js files, "
            f"found {len(mod_files)}: {mod_files}"
        )

    def test_manifest_matches_disk(self) -> None:
        """Every file in the manifest exists on disk, and vice versa."""
        static_dir = Path(__file__).resolve().parent.parent / "app" / "static"
        on_disk = sorted(p.name for p in static_dir.glob("mod-*.js"))
        in_manifest = sorted(MODULE_MANIFEST.keys())
        assert on_disk == in_manifest, (
            f"Mismatch between disk and manifest.\n"
            f"  On disk only: {set(on_disk) - set(in_manifest)}\n"
            f"  In manifest only: {set(in_manifest) - set(on_disk)}"
        )
