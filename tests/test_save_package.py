"""
tests/test_save_package.py
─────────────────────────────────────────────────────────────────────────────
Tests for app/save_package.py — manifest, zip export/import, text extraction.

Each test class targets a specific public or private function:

1. ``_compute_file_sha256`` — determinism and hex format.
2. ``build_manifest`` — structure, roles, checksums, version.
3. ``create_zip_archive`` — valid zip, correct content, filtering.
4. ``validate_and_extract_zip`` — happy path, security limits, checksums.
5. ``extract_body_text`` — header stripping for output.md / baseline.md.
6. ``extract_fenced_code`` — fence extraction for system_prompt.md.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from app.save_package import (
    _compute_file_sha256,
    _validate_checksums,
    build_manifest,
    create_zip_archive,
    extract_body_text,
    extract_fenced_code,
    validate_and_extract_zip,
)

# ─────────────────────────────────────────────────────────────────────────────
# _compute_file_sha256
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeFileSha256:
    """Verify the per-file SHA-256 helper."""

    def test_returns_64_char_hex_string(self, tmp_path: Path) -> None:
        """The digest must be a 64-character lowercase hex string."""
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        result = _compute_file_sha256(f)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_determinism(self, tmp_path: Path) -> None:
        """Same file content must always produce the same hash."""
        f = tmp_path / "test.txt"
        f.write_text("deterministic content", encoding="utf-8")
        assert _compute_file_sha256(f) == _compute_file_sha256(f)

    def test_sensitivity(self, tmp_path: Path) -> None:
        """Different content must produce different hashes."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A", encoding="utf-8")
        f2.write_text("content B", encoding="utf-8")
        assert _compute_file_sha256(f1) != _compute_file_sha256(f2)

    def test_matches_hashlib_directly(self, tmp_path: Path) -> None:
        """The result must match a manual hashlib.sha256() computation."""
        f = tmp_path / "test.txt"
        content = "verify against hashlib"
        f.write_text(content, encoding="utf-8")
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert _compute_file_sha256(f) == expected


# ─────────────────────────────────────────────────────────────────────────────
# build_manifest
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildManifest:
    """Verify manifest construction from a save folder."""

    def _write_save_files(self, save_dir: Path) -> list[str]:
        """Helper: write a minimal set of save files and return the filename list."""
        files = ["payload.json", "system_prompt.md"]
        (save_dir / "payload.json").write_text('{"axes": {}}', encoding="utf-8")
        (save_dir / "system_prompt.md").write_text(
            "# System Prompt\n\n```text\nTest\n```\n", encoding="utf-8"
        )
        return files

    def test_manifest_version_is_one(self, tmp_path: Path) -> None:
        """The manifest must include manifest_version=1."""
        files = self._write_save_files(tmp_path)
        manifest = build_manifest(tmp_path, files)
        assert manifest["manifest_version"] == 1

    def test_all_files_listed(self, tmp_path: Path) -> None:
        """Every file in files_written plus metadata.json must appear."""
        files = self._write_save_files(tmp_path)
        manifest = build_manifest(tmp_path, files)
        assert "payload.json" in manifest["files"]
        assert "system_prompt.md" in manifest["files"]
        assert "metadata.json" in manifest["files"]

    def test_metadata_json_has_null_sha256(self, tmp_path: Path) -> None:
        """metadata.json cannot hash itself — sha256 must be None."""
        files = self._write_save_files(tmp_path)
        manifest = build_manifest(tmp_path, files)
        assert manifest["files"]["metadata.json"]["sha256"] is None

    def test_metadata_json_has_zero_size(self, tmp_path: Path) -> None:
        """metadata.json size_bytes is 0 (sentinel — not yet written)."""
        files = self._write_save_files(tmp_path)
        manifest = build_manifest(tmp_path, files)
        assert manifest["files"]["metadata.json"]["size_bytes"] == 0

    def test_other_files_have_non_null_sha256(self, tmp_path: Path) -> None:
        """All files except metadata.json must have a real SHA-256 hash."""
        files = self._write_save_files(tmp_path)
        manifest = build_manifest(tmp_path, files)
        for name in files:
            entry = manifest["files"][name]
            assert entry["sha256"] is not None
            assert len(entry["sha256"]) == 64

    def test_roles_are_correct(self, tmp_path: Path) -> None:
        """Each file must be assigned the correct role from _FILE_ROLES."""
        files = self._write_save_files(tmp_path)
        manifest = build_manifest(tmp_path, files)
        assert manifest["files"]["payload.json"]["role"] == "payload"
        assert manifest["files"]["system_prompt.md"]["role"] == "system_prompt"
        assert manifest["files"]["metadata.json"]["role"] == "provenance"

    def test_size_bytes_matches_actual(self, tmp_path: Path) -> None:
        """size_bytes must match the actual file size on disk."""
        files = self._write_save_files(tmp_path)
        manifest = build_manifest(tmp_path, files)
        for name in files:
            expected_size = (tmp_path / name).stat().st_size
            assert manifest["files"][name]["size_bytes"] == expected_size

    def test_checksum_matches_file_content(self, tmp_path: Path) -> None:
        """The SHA-256 in the manifest must match re-reading the file."""
        files = self._write_save_files(tmp_path)
        manifest = build_manifest(tmp_path, files)
        for name in files:
            actual = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            assert manifest["files"][name]["sha256"] == actual


# ─────────────────────────────────────────────────────────────────────────────
# create_zip_archive
# ─────────────────────────────────────────────────────────────────────────────


class TestCreateZipArchive:
    """Verify zip archive creation from a save folder."""

    def _populate_save_dir(self, save_dir: Path) -> list[str]:
        """Helper: write typical save files and return the expected filenames."""
        filenames = ["metadata.json", "payload.json", "system_prompt.md"]
        for name in filenames:
            (save_dir / name).write_text(f"content of {name}", encoding="utf-8")
        return filenames

    def test_returns_valid_zip_bytes(self, tmp_path: Path) -> None:
        """The returned bytes must be a valid zip file."""
        self._populate_save_dir(tmp_path)
        result = create_zip_archive(tmp_path)
        assert zipfile.is_zipfile(io.BytesIO(result))

    def test_all_known_files_present(self, tmp_path: Path) -> None:
        """Every known-role file in the folder must appear in the zip."""
        expected = self._populate_save_dir(tmp_path)
        result = create_zip_archive(tmp_path)
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            names = zf.namelist()
        for name in expected:
            assert name in names

    def test_unknown_files_excluded(self, tmp_path: Path) -> None:
        """Files not in _FILE_ROLES (e.g. .DS_Store) must be excluded."""
        self._populate_save_dir(tmp_path)
        (tmp_path / ".DS_Store").write_bytes(b"\x00\x00")
        (tmp_path / "random.txt").write_text("not a save file", encoding="utf-8")
        result = create_zip_archive(tmp_path)
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            names = zf.namelist()
        assert ".DS_Store" not in names
        assert "random.txt" not in names

    def test_content_round_trips(self, tmp_path: Path) -> None:
        """Extracted file content must match the original file content."""
        self._populate_save_dir(tmp_path)
        result = create_zip_archive(tmp_path)
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            for name in zf.namelist():
                original = (tmp_path / name).read_bytes()
                assert zf.read(name) == original

    def test_flat_filenames_no_directories(self, tmp_path: Path) -> None:
        """Zip entries must use flat names — no directory prefixes."""
        self._populate_save_dir(tmp_path)
        result = create_zip_archive(tmp_path)
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            for name in zf.namelist():
                assert "/" not in name
                assert "\\" not in name


# ─────────────────────────────────────────────────────────────────────────────
# validate_and_extract_zip
# ─────────────────────────────────────────────────────────────────────────────


def _make_zip(files: dict[str, bytes]) -> bytes:
    """Helper: create a zip archive in memory from a {name: bytes} dict."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestValidateAndExtractZip:
    """Verify zip import validation and extraction."""

    def test_happy_path_with_manifest(self) -> None:
        """Valid zip with manifest: all files extracted, no warnings, checksums pass."""
        payload_bytes = b'{"axes": {}}'
        prompt_bytes = b"# System Prompt\n\n```text\nTest\n```\n"

        # Build a metadata.json that includes a manifest with correct checksums
        manifest = {
            "manifest_version": 1,
            "files": {
                "payload.json": {
                    "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                    "role": "payload",
                    "size_bytes": len(payload_bytes),
                },
                "system_prompt.md": {
                    "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                    "role": "system_prompt",
                    "size_bytes": len(prompt_bytes),
                },
                "metadata.json": {
                    "sha256": None,
                    "role": "provenance",
                    "size_bytes": 0,
                },
            },
        }
        metadata = {"folder_name": "test", "manifest": manifest}
        metadata_bytes = json.dumps(metadata).encode("utf-8")

        zip_bytes = _make_zip(
            {
                "metadata.json": metadata_bytes,
                "payload.json": payload_bytes,
                "system_prompt.md": prompt_bytes,
            }
        )

        files, warnings = validate_and_extract_zip(zip_bytes)
        assert "metadata.json" in files
        assert "payload.json" in files
        assert "system_prompt.md" in files
        assert len(warnings) == 0

    def test_happy_path_without_manifest(self) -> None:
        """Valid zip without manifest: files extracted, warning about missing manifest."""
        metadata = {"folder_name": "test"}  # No manifest key
        zip_bytes = _make_zip(
            {
                "metadata.json": json.dumps(metadata).encode("utf-8"),
                "payload.json": b'{"axes": {}}',
            }
        )

        files, warnings = validate_and_extract_zip(zip_bytes)
        assert "payload.json" in files
        assert any("No manifest" in w for w in warnings)

    def test_no_metadata_json_warns(self) -> None:
        """Zip without metadata.json: files extracted, warning about missing metadata."""
        zip_bytes = _make_zip({"payload.json": b'{"axes": {}}'})
        files, warnings = validate_and_extract_zip(zip_bytes)
        assert "payload.json" in files
        assert any("No metadata.json" in w for w in warnings)

    def test_checksum_mismatch_raises(self) -> None:
        """If a file's SHA-256 doesn't match the manifest, ValueError is raised."""
        payload_bytes = b'{"axes": {}}'
        manifest = {
            "manifest_version": 1,
            "files": {
                "payload.json": {
                    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "role": "payload",
                    "size_bytes": len(payload_bytes),
                },
                "metadata.json": {"sha256": None, "role": "provenance", "size_bytes": 0},
            },
        }
        metadata = {"manifest": manifest}
        zip_bytes = _make_zip(
            {
                "metadata.json": json.dumps(metadata).encode("utf-8"),
                "payload.json": payload_bytes,
            }
        )

        with pytest.raises(ValueError, match="Checksum mismatch"):
            validate_and_extract_zip(zip_bytes)

    def test_not_a_zip_raises(self) -> None:
        """Non-zip bytes must raise ValueError."""
        with pytest.raises(ValueError, match="not a valid zip"):
            validate_and_extract_zip(b"this is not a zip file")

    def test_path_traversal_rejected(self) -> None:
        """Zip entries with path separators must raise ValueError."""
        zip_bytes = _make_zip({"../../etc/passwd": b"root:x:0:0"})
        with pytest.raises(ValueError, match="path separators"):
            validate_and_extract_zip(zip_bytes)

    def test_unknown_files_skipped_with_warning(self) -> None:
        """Files not in _FILE_ROLES are skipped and a warning is emitted."""
        zip_bytes = _make_zip(
            {
                "metadata.json": json.dumps({"folder_name": "test"}).encode(),
                "unknown_file.txt": b"hello",
            }
        )
        files, warnings = validate_and_extract_zip(zip_bytes)
        assert "unknown_file.txt" not in files
        assert any("Skipped unknown" in w for w in warnings)

    def test_too_many_files_raises(self) -> None:
        """Zips with more than MAX_FILE_COUNT entries must raise ValueError."""
        # Create a zip with 21 entries (all with names that won't match
        # _FILE_ROLES, but the count check happens before name filtering)
        many_files = {f"file_{i}.txt": b"data" for i in range(21)}
        zip_bytes = _make_zip(many_files)
        with pytest.raises(ValueError, match="exceeding the maximum"):
            validate_and_extract_zip(zip_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# _validate_checksums
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateChecksums:
    """Verify the checksum validation helper directly."""

    def test_valid_checksums_pass_silently(self) -> None:
        """Correct checksums should not raise any error."""
        data = b"hello world"
        extracted = {"payload.json": data}
        manifest = {
            "files": {
                "payload.json": {"sha256": hashlib.sha256(data).hexdigest()},
                "metadata.json": {"sha256": None},
            }
        }
        # Should not raise
        _validate_checksums(extracted, manifest)

    def test_missing_file_in_extracted_is_tolerated(self) -> None:
        """Files listed in manifest but absent from zip are tolerated."""
        manifest = {
            "files": {
                "output.md": {"sha256": "abc123"},
                "metadata.json": {"sha256": None},
            }
        }
        # output.md not in extracted — should not raise
        _validate_checksums({}, manifest)

    def test_mismatch_raises_value_error(self) -> None:
        """A wrong checksum must raise ValueError with the filename."""
        extracted = {"payload.json": b"actual content"}
        manifest = {
            "files": {
                "payload.json": {"sha256": "0" * 64},
            }
        }
        with pytest.raises(ValueError, match="payload.json"):
            _validate_checksums(extracted, manifest)


# ─────────────────────────────────────────────────────────────────────────────
# extract_body_text
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractBodyText:
    """Verify header stripping for output.md and baseline.md."""

    def test_strips_heading_and_comments(self) -> None:
        """Standard output.md format: heading + comments → body text returned."""
        content = (
            "# Output\n"
            "\n"
            "<!-- Axis Descriptor Lab – generated output -->\n"
            "<!-- saved: 2026-02-19T09:29:22 -->\n"
            "<!-- model: gemma2:2b | temp: 0.2 -->\n"
            "\n"
            "The weathered figure stands near the threshold."
        )
        result = extract_body_text(content)
        assert result == "The weathered figure stands near the threshold."

    def test_baseline_md_format(self) -> None:
        """Standard baseline.md format: heading + comment → body text."""
        content = (
            "# Baseline (A)\n"
            "\n"
            "<!-- Axis Descriptor Lab – baseline text for save test -->\n"
            "\n"
            "A dark goblin lurks beyond the gate."
        )
        result = extract_body_text(content)
        assert result == "A dark goblin lurks beyond the gate."

    def test_multiline_body(self) -> None:
        """Multi-line body text should be preserved."""
        content = (
            "# Output\n" "\n" "<!-- comment -->\n" "\n" "Line one.\n" "Line two.\n" "Line three."
        )
        result = extract_body_text(content)
        assert "Line one." in result
        assert "Line two." in result
        assert "Line three." in result

    def test_no_header_returns_full_text(self) -> None:
        """If there's no heading or comment, the full text is returned."""
        content = "Just plain text, no headers."
        result = extract_body_text(content)
        assert result == "Just plain text, no headers."

    def test_only_headers_returns_content(self) -> None:
        """If the file is all headers with no body, return stripped content."""
        content = "# Heading\n\n<!-- comment -->\n"
        result = extract_body_text(content)
        # Falls back to returning the full content stripped
        assert result == content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# extract_fenced_code
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractFencedCode:
    """Verify fenced code block extraction for system_prompt.md."""

    def test_extracts_text_fence(self) -> None:
        """Standard system_prompt.md with ```text fence."""
        content = (
            "# System Prompt\n"
            "\n"
            "<!-- Axis Descriptor Lab -->\n"
            "\n"
            "```text\n"
            "You are a descriptive layer.\n"
            "Do not mention the policy hash.\n"
            "```\n"
        )
        result = extract_fenced_code(content)
        assert result == "You are a descriptive layer.\nDo not mention the policy hash."

    def test_extracts_bare_fence(self) -> None:
        """Fenced block with ``` (no language tag) should also work."""
        content = "```\nSome prompt text.\n```"
        result = extract_fenced_code(content)
        assert result == "Some prompt text."

    def test_no_fence_falls_back_to_body_text(self) -> None:
        """If no fenced block exists, fall back to extract_body_text()."""
        content = "# System Prompt\n" "\n" "<!-- comment -->\n" "\n" "Unfenced prompt text here."
        result = extract_fenced_code(content)
        assert result == "Unfenced prompt text here."

    def test_preserves_internal_whitespace(self) -> None:
        """Relative indentation inside the fenced block should be preserved."""
        content = "```text\nline one\n    indented line\nline three\n```"
        result = extract_fenced_code(content)
        assert "line one" in result
        assert "    indented line" in result
        assert "line three" in result

    def test_plain_text_without_headers(self) -> None:
        """Plain text with no markdown at all returns as-is."""
        content = "Just raw prompt text."
        result = extract_fenced_code(content)
        assert result == "Just raw prompt text."
