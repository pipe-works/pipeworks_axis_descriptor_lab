"""Unit tests for world/lab/legacy path resolution precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.path_resolver import (
    PathResolutionError,
    resolve_axis_payload_paths,
    resolve_lexicon_paths,
    resolve_policy_bundle_paths,
    resolve_prompt_paths,
)


def _write(path: Path, content: str) -> None:
    """Create one UTF-8 text file with parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestResolvePromptPaths:
    """Prompt resolution precedence and ambiguity checks."""

    def test_prefers_world_prompt_over_legacy_for_chat_translation(self, tmp_path: Path) -> None:
        world_root = tmp_path / "worlds"
        lab_root = tmp_path / "lab_only"
        legacy_root = tmp_path / "legacy_prompts"
        _write(
            world_root
            / "pipeworks_web"
            / "policies"
            / "translation"
            / "prompts"
            / "ic"
            / "pipeworks_web_ic_prompt.txt",
            "world prompt",
        )
        _write(
            legacy_root / "chat_translation" / "pipeworks_web_ic_prompt.txt",
            "legacy prompt",
        )

        resolved = resolve_prompt_paths(
            "chat_translation",
            world_id="pipeworks_web",
            world_root=world_root,
            lab_only_root=lab_root,
            legacy_prompts_root=legacy_root,
        )["pipeworks_web_ic_prompt"]

        assert resolved.source_kind == "world_canonical"
        assert resolved.source_path == "policies/translation/prompts/ic/pipeworks_web_ic_prompt.txt"

    def test_prefers_target_lab_only_over_transitional_for_chat_translation(
        self, tmp_path: Path
    ) -> None:
        world_root = tmp_path / "worlds"
        lab_root = tmp_path / "lab_only"
        legacy_root = tmp_path / "legacy_prompts"
        _write(
            lab_root / "prompts" / "chat_translation" / "pipeworks_web_ic_prompt.txt",
            "new lab path",
        )
        _write(
            lab_root / "chat_translation" / "prompts" / "pipeworks_web_ic_prompt.txt",
            "old lab path",
        )
        _write(legacy_root / "chat_translation" / "pipeworks_web_ic_prompt.txt", "legacy")

        resolved = resolve_prompt_paths(
            "chat_translation",
            world_id="pipeworks_web",
            world_root=world_root,
            lab_only_root=lab_root,
            legacy_prompts_root=legacy_root,
        )["pipeworks_web_ic_prompt"]

        assert resolved.source_kind == "lab_only"
        assert resolved.path.read_text(encoding="utf-8") == "new lab path"

    def test_prefers_target_lab_only_over_legacy_for_character_description(
        self, tmp_path: Path
    ) -> None:
        world_root = tmp_path / "worlds"
        lab_root = tmp_path / "lab_only"
        legacy_root = tmp_path / "legacy_prompts"
        _write(
            lab_root / "prompts" / "character_description" / "system_prompt_v01.txt",
            "lab prompt",
        )
        _write(
            legacy_root / "character_description" / "system_prompt_v01.txt",
            "legacy prompt",
        )

        resolved = resolve_prompt_paths(
            "character_description",
            world_id="pipeworks_web",
            world_root=world_root,
            lab_only_root=lab_root,
            legacy_prompts_root=legacy_root,
        )["system_prompt_v01"]

        assert resolved.source_kind == "lab_only"
        assert resolved.source_path == "system_prompt_v01.txt"

    def test_duplicate_prompt_names_same_precedence_raise(self, tmp_path: Path) -> None:
        legacy_root = tmp_path / "legacy_prompts"
        _write(legacy_root / "chat_translation" / "duplicate.txt", "a")
        _write(legacy_root / "chat_translation" / "drafts" / "duplicate.txt", "b")

        with pytest.raises(PathResolutionError):
            resolve_prompt_paths(
                "chat_translation",
                world_id="pipeworks_web",
                world_root=tmp_path / "worlds",
                lab_only_root=tmp_path / "lab_only",
                legacy_prompts_root=legacy_root,
            )


class TestResolveAxisPayloadPaths:
    """Axis payload precedence checks."""

    def test_prefers_target_lab_only_examples_over_transitional_and_legacy(
        self, tmp_path: Path
    ) -> None:
        world_root = tmp_path / "worlds"
        lab_root = tmp_path / "lab_only"
        legacy_root = tmp_path / "legacy_examples"
        _write(
            lab_root / "axis" / "examples" / "proud_operator.json",
            '{"axes":{"demeanor":{"label":"new-lab","score":0.8}},"policy_hash":"a","seed":1,"world_id":"pipeworks_web"}',
        )
        _write(
            lab_root / "examples" / "proud_operator.json",
            '{"axes":{"demeanor":{"label":"old-lab","score":0.7}},"policy_hash":"a","seed":1,"world_id":"pipeworks_web"}',
        )
        _write(
            legacy_root / "proud_operator.json",
            '{"axes":{"demeanor":{"label":"legacy","score":0.2}},"policy_hash":"b","seed":2,"world_id":"pipeworks_web"}',
        )

        resolved = resolve_axis_payload_paths(
            world_id="pipeworks_web",
            world_root=world_root,
            lab_only_root=lab_root,
            legacy_examples_root=legacy_root,
        )["proud_operator"]

        assert resolved.source_kind == "lab_only"
        assert resolved.path.read_text(encoding="utf-8").find('"new-lab"') >= 0


class TestResolveLexiconPaths:
    """Lexicon resolution precedence checks."""

    def test_prefers_target_lab_only_lexicons_over_transitional_and_legacy(
        self, tmp_path: Path
    ) -> None:
        lab_root = tmp_path / "lab_only"
        legacy_root = tmp_path / "legacy_lexicons"
        _write(lab_root / "axis" / "lexicons" / "abstraction_v0_1.json", '{"version":"0.1"}')
        _write(lab_root / "lexicons" / "abstraction_v0_1.json", '{"version":"0.0"}')
        _write(legacy_root / "abstraction_v0_1.json", '{"version":"legacy"}')

        resolved = resolve_lexicon_paths(
            lab_only_root=lab_root,
            legacy_lexicons_root=legacy_root,
        )["abstraction_v0_1"]

        assert resolved.source_kind == "lab_only"
        assert resolved.path.read_text(encoding="utf-8") == '{"version":"0.1"}'


class TestResolvePolicyBundlePaths:
    """Policy bundle precedence checks."""

    def test_prefers_world_draft_policy_bundle(self, tmp_path: Path) -> None:
        world_root = tmp_path / "worlds"
        lab_root = tmp_path / "lab_only"
        legacy_root = tmp_path / "legacy_policy_bundles"
        _write(
            world_root
            / "pipeworks_web"
            / "policies"
            / "drafts"
            / "policy_bundles"
            / "pipeworks_web_policy_bundle_v0_1.json",
            "{}",
        )
        _write(
            legacy_root / "pipeworks_web_policy_bundle_v0_1.json",
            "{}",
        )

        resolved = resolve_policy_bundle_paths(
            world_id="pipeworks_web",
            world_root=world_root,
            lab_only_root=lab_root,
            legacy_policy_bundle_root=legacy_root,
        )["pipeworks_web_policy_bundle_v0_1"]

        assert resolved.source_kind == "world_draft"
        assert (
            resolved.source_path
            == "policies/drafts/policy_bundles/pipeworks_web_policy_bundle_v0_1.json"
        )

    def test_prefers_target_lab_only_policy_bundles_over_transitional(self, tmp_path: Path) -> None:
        world_root = tmp_path / "worlds"
        lab_root = tmp_path / "lab_only"
        legacy_root = tmp_path / "legacy_policy_bundles"
        _write(lab_root / "policy_bundles" / "preview.json", '{"version":"new"}')
        _write(lab_root / "policy_bundle_previews" / "preview.json", '{"version":"old"}')
        _write(legacy_root / "preview.json", '{"version":"legacy"}')

        resolved = resolve_policy_bundle_paths(
            world_id="pipeworks_web",
            world_root=world_root,
            lab_only_root=lab_root,
            legacy_policy_bundle_root=legacy_root,
        )["preview"]

        assert resolved.source_kind == "lab_only"
        assert resolved.path.read_text(encoding="utf-8") == '{"version":"new"}'
