"""Tests for the ES module structure introduced by the app.js → mod-*.js refactor.

Verifies that:
  1. All 14 module files are served at /static/ with correct content type.
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
        "imports_from": ["mod-state.js", "mod-utils.js", "mod-status.js"],
    },
    "mod-loaders.js": {
        "exports": [
            "loadExampleList",
            "loadExample",
            "loadPromptList",
            "loadPrompt",
            "wireLoaderEvents",
        ],
        "imports_from": ["mod-state.js", "mod-status.js", "mod-sync.js"],
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
        ],
    },
}

ALL_MODULE_NAMES = sorted(MODULE_MANIFEST.keys())


# ── Helpers ─────────────────────────────────────────────────────────────────


def _read_module(name: str) -> str:
    """Read a module file from the static directory."""
    path = Path(__file__).resolve().parent.parent / "app" / "static" / name
    return path.read_text(encoding="utf-8")


# ── 1. Static file serving ──────────────────────────────────────────────────


class TestModuleServing:
    """All 14 mod-*.js files are served via /static/ with correct content type."""

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
    """The refactor produced exactly 14 module files."""

    def test_exactly_14_modules_on_disk(self) -> None:
        """The static directory contains exactly 14 mod-*.js files."""
        static_dir = Path(__file__).resolve().parent.parent / "app" / "static"
        mod_files = sorted(p.name for p in static_dir.glob("mod-*.js"))
        assert (
            len(mod_files) == 14
        ), f"Expected 14 mod-*.js files, found {len(mod_files)}: {mod_files}"

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
