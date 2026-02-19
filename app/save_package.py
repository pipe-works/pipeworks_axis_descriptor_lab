"""
app/save_package.py
-----------------------------------------------------------------------------
Save package utilities for the Axis Descriptor Lab.

Why a dedicated module?
-----------------------
The save system writes multiple files into a timestamped folder.  Making these
packages self-describing (manifest with checksums), portable (zip export), and
re-importable (zip upload → state restoration) requires logic that is
independent of HTTP routing and Pydantic schemas.  Keeping it here avoids
bloating ``main.py`` and follows the one-module-per-responsibility pattern
established by ``hashing.py``, ``signal_isolation.py``, and
``transformation_map.py``.

Sections
--------
1. **Manifest construction** — build a ``manifest`` dict with per-file
   SHA-256 checksums, roles, and byte sizes.
2. **Zip export** — bundle a save folder into a compressed zip archive.
3. **Zip import** — validate and extract an uploaded zip, verifying manifest
   checksums for scientific integrity.
4. **Markdown text extraction** — strip provenance headers from ``output.md``,
   ``baseline.md``, and ``system_prompt.md`` to recover plain text for
   frontend state restoration.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps each known save-package filename to a human-readable role string.
# Used by the manifest builder and the zip importer to identify and filter
# files.  Any file whose name is not in this dict is ignored during import.
_FILE_ROLES: dict[str, str] = {
    "metadata.json": "provenance",
    "payload.json": "payload",
    "system_prompt.md": "system_prompt",
    "output.md": "output",
    "baseline.md": "baseline",
    "delta.json": "delta",
    "transformation_map.json": "transformation_map",
}

# Security limits for zip import.  These prevent zip bombs and excessively
# large uploads from consuming server resources.
MAX_FILE_SIZE: int = 5_242_880  # 5 MB per individual file inside the zip
MAX_FILE_COUNT: int = 20  # maximum number of entries allowed in the zip
MAX_UPLOAD_SIZE: int = 10_485_760  # 10 MB total upload size


# ---------------------------------------------------------------------------
# Section 1: Manifest construction
# ---------------------------------------------------------------------------


def _compute_file_sha256(path: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file's raw bytes.

    Reads the file in a single pass (save packages are small — typically
    under 10 KB total) and returns the 64-character lowercase hex string.

    Parameters
    ----------
    path : Absolute or relative path to the file to hash.

    Returns
    -------
    str : 64-character lowercase hexadecimal SHA-256 digest.
    """
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def build_manifest(save_dir: Path, files_written: list[str]) -> dict:
    """
    Build a manifest dict describing every file in a save package.

    The manifest provides scientific integrity metadata: each file gets a
    SHA-256 checksum, a role classification, and a byte size.  This allows
    downstream consumers (import, audit tools) to verify that no file has
    been tampered with or corrupted.

    ``metadata.json`` is a special case: it cannot hash itself (the hash
    would change the content, which would change the hash — an infinite
    loop).  Its entry uses ``sha256: null`` and ``size_bytes: 0`` as
    sentinel values.

    Parameters
    ----------
    save_dir       : Path to the save folder containing the written files.
    files_written  : Ordered list of filenames already written to ``save_dir``
                     (not including ``metadata.json``, which is written after
                     the manifest is built).

    Returns
    -------
    dict : Manifest dict with ``manifest_version`` (int) and ``files``
           (dict mapping filename → {sha256, role, size_bytes}).
    """
    files_manifest: dict[str, dict] = {}

    for filename in files_written:
        file_path = save_dir / filename
        files_manifest[filename] = {
            "sha256": _compute_file_sha256(file_path),
            "role": _FILE_ROLES.get(filename, "unknown"),
            "size_bytes": file_path.stat().st_size,
        }

    # metadata.json cannot hash itself — use null sentinel values.
    # It will be written AFTER the manifest is embedded in its content.
    files_manifest["metadata.json"] = {
        "sha256": None,
        "role": "provenance",
        "size_bytes": 0,
    }

    return {
        "manifest_version": 1,
        "files": files_manifest,
    }


# ---------------------------------------------------------------------------
# Section 2: Zip export
# ---------------------------------------------------------------------------


def create_zip_archive(save_dir: Path) -> bytes:
    """
    Bundle a save folder into a compressed zip archive.

    Only files whose names appear in ``_FILE_ROLES`` are included — any
    unexpected files (e.g. OS metadata like ``.DS_Store``) are silently
    skipped.  Files are stored with flat names (no directory nesting) so
    the zip extracts cleanly into a single folder.

    Parameters
    ----------
    save_dir : Path to the save folder to archive.

    Returns
    -------
    bytes : Raw zip file bytes, ready to stream to the client or write
            to disk.
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for child in sorted(save_dir.iterdir()):
            # Only include files with known roles — skip directories and
            # unexpected files (e.g. .DS_Store, __pycache__).
            if child.is_file() and child.name in _FILE_ROLES:
                zf.write(child, arcname=child.name)

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Section 3: Zip import — validation and extraction
# ---------------------------------------------------------------------------


def validate_and_extract_zip(
    zip_bytes: bytes,
) -> tuple[dict[str, bytes], list[str]]:
    """
    Parse, validate, and extract a save-package zip archive.

    Performs security validation first (size limits, path traversal, known
    filenames), then optionally validates manifest checksums if a manifest
    is present in ``metadata.json``.

    Parameters
    ----------
    zip_bytes : Raw bytes of the uploaded zip file.

    Returns
    -------
    tuple[dict[str, bytes], list[str]]
        A 2-tuple of:
        - ``files``: dict mapping filename → raw file bytes for every
          valid entry extracted from the zip.
        - ``warnings``: list of non-fatal warning strings (e.g. "No
          manifest found in metadata.json — checksums not verified").

    Raises
    ------
    ValueError
        If the input is not a valid zip, exceeds size limits, contains
        path-traversal entries, or has manifest checksum mismatches.
    """
    warnings: list[str] = []

    # -- Gate: is it a valid zip file? ------------------------------------ #
    if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
        raise ValueError("Uploaded file is not a valid zip archive.")

    extracted: dict[str, bytes] = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
        entries = zf.infolist()

        # -- Security: limit the number of entries ----------------------- #
        if len(entries) > MAX_FILE_COUNT:
            raise ValueError(
                f"Zip contains {len(entries)} entries, exceeding the "
                f"maximum of {MAX_FILE_COUNT}."
            )

        for info in entries:
            name = info.filename

            # -- Security: reject path traversal attempts ---------------- #
            # Any entry with directory separators or parent references is
            # suspicious and rejected outright.
            if "/" in name or "\\" in name or ".." in name:
                raise ValueError(
                    f"Zip entry '{name}' contains path separators or "
                    f"parent references — possible path traversal."
                )

            # -- Security: only accept known save-package filenames ------ #
            if name not in _FILE_ROLES:
                warnings.append(f"Skipped unknown file '{name}'.")
                continue

            # -- Security: enforce per-file size limit ------------------- #
            if info.file_size > MAX_FILE_SIZE:
                raise ValueError(
                    f"Zip entry '{name}' is {info.file_size:,} bytes, "
                    f"exceeding the {MAX_FILE_SIZE:,}-byte limit."
                )

            extracted[name] = zf.read(name)

    # -- Validate manifest checksums (if present) ------------------------- #
    if "metadata.json" in extracted:
        try:
            metadata = json.loads(extracted["metadata.json"].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"metadata.json is not valid JSON: {exc}") from exc

        manifest = metadata.get("manifest")
        if manifest is not None:
            _validate_checksums(extracted, manifest)
        else:
            warnings.append("No manifest found in metadata.json — checksums not verified.")
    else:
        warnings.append("No metadata.json found in zip — checksums not verified.")

    return extracted, warnings


def _validate_checksums(
    extracted: dict[str, bytes],
    manifest: dict,
) -> None:
    """
    Verify SHA-256 checksums from the manifest against extracted file bytes.

    Iterates over every file listed in the manifest's ``files`` dict.  For
    each entry with a non-null ``sha256``, computes the SHA-256 of the
    corresponding extracted bytes and compares.  Raises ``ValueError`` on
    the first mismatch.

    ``metadata.json`` has ``sha256: null`` in the manifest (it cannot hash
    itself) and is always skipped.

    Parameters
    ----------
    extracted : Dict of filename → raw bytes from the zip.
    manifest  : The ``manifest`` dict from metadata.json, containing
                ``files`` with per-file ``sha256`` values.

    Raises
    ------
    ValueError : If any file's computed SHA-256 does not match the manifest.
    """
    manifest_files = manifest.get("files", {})

    for filename, entry in manifest_files.items():
        expected_hash = entry.get("sha256")

        # metadata.json has sha256=null — skip it.
        if expected_hash is None:
            continue

        if filename not in extracted:
            # File listed in manifest but not present in zip — this is
            # acceptable for optional files (output.md, baseline.md, etc.)
            continue

        actual_hash = hashlib.sha256(extracted[filename]).hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch for '{filename}': "
                f"expected {expected_hash[:16]}…, "
                f"got {actual_hash[:16]}…"
            )


# ---------------------------------------------------------------------------
# Section 4: Markdown text extraction
# ---------------------------------------------------------------------------


def extract_body_text(content: str) -> str:
    """
    Strip the Markdown heading and HTML comment header from ``output.md``
    or ``baseline.md``, returning only the body paragraph.

    The save system writes these files with a structure like::

        # Output

        <!-- Axis Descriptor Lab – generated output -->
        <!-- saved: 2026-02-19T09:29:22 -->
        <!-- model: gemma2:2b | temp: 0.2 | ... -->

        The actual generated text starts here...

    This function walks the lines, skipping the heading (``# ...``), blank
    lines, and HTML comments (``<!-- ... -->``) until it reaches the first
    line of body text.  Everything from that point onward is returned.

    Parameters
    ----------
    content : The full text of an ``output.md`` or ``baseline.md`` file.

    Returns
    -------
    str : The body text with leading/trailing whitespace stripped.  If no
          body text is found (e.g. the file is all headers), the original
          content is returned as a fallback.
    """
    lines = content.split("\n")
    body_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip blank lines, markdown headings, and HTML comments.
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        # Multi-line HTML comments: skip lines that start with <!--
        # but don't close on the same line (unlikely in our format,
        # but handled for robustness).
        if stripped.startswith("<!--"):
            continue

        # First non-header, non-comment, non-blank line = body start.
        body_start = i
        break

    if body_start is None:
        # Fallback: no body text found — return the whole thing.
        return content.strip()

    return "\n".join(lines[body_start:]).strip()


def extract_fenced_code(content: str) -> str:
    """
    Extract text from a fenced code block in ``system_prompt.md``.

    The save system writes the system prompt wrapped in a Markdown fenced
    code block::

        # System Prompt

        <!-- Axis Descriptor Lab – system prompt for save ... -->

        ```text
        You are a descriptive layer inside a deterministic system.
        ...
        ```

    This function finds the opening fence (a line starting with `` ```text ``
    or just `` ``` ``) and the closing fence (a line that is exactly `` ``` ``),
    and returns everything between them.

    Parameters
    ----------
    content : The full text of a ``system_prompt.md`` file.

    Returns
    -------
    str : The extracted prompt text with leading/trailing whitespace stripped.
          If no fenced code block is found, falls back to
          ``extract_body_text()`` to strip headers and return the body.
    """
    lines = content.split("\n")
    inside_fence = False
    fence_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not inside_fence:
            # Look for the opening fence: ```text or just ```
            if stripped.startswith("```"):
                inside_fence = True
                continue
        else:
            # Look for the closing fence: exactly ```
            if stripped == "```":
                break
            fence_lines.append(line)

    if fence_lines:
        return "\n".join(fence_lines).strip()

    # Fallback: no fenced code block found — try body text extraction.
    return extract_body_text(content)
