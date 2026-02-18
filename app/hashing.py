"""
app/hashing.py
─────────────────────────────────────────────────────────────────────────────
Consolidated hashing utilities for the Axis Descriptor Lab.

Why a dedicated module?
───────────────────────
Hashing logic is used across multiple routes (generate, log, save) and must
produce identical results regardless of call site.  Centralising normalisation
rules and hash functions in one place eliminates duplication and ensures that
a single change to normalisation propagates everywhere.

Hash types
──────────
This module provides four public hash functions:

1. ``compute_payload_hash``       – SHA-256 of the canonical AxisPayload JSON.
2. ``compute_system_prompt_hash`` – SHA-256 of the normalised system prompt.
3. ``compute_output_hash``        – SHA-256 of the normalised LLM output text.
4. ``compute_ipc_id``             – SHA-256 of the concatenated provenance
                                    fields (the Interpretive Provenance Chain
                                    identifier).

All functions return lowercase 64-character hex digest strings.

Normalisation philosophy
────────────────────────
Raw text is normalised before hashing so that semantically identical content
produces the same digest even when formatting differs.  The rules are
intentionally conservative:

- **Case is never changed** — upper/lowercase carries semantic meaning in both
  prompts and generated text.
- **Internal structure is preserved** — line order, sentence order, and
  punctuation are never altered.
- **Only noise is removed** — leading/trailing whitespace, extra spaces, and
  blank padding lines at the edges of the text.

This ensures that meaningful edits always change the hash, while trivial
formatting differences (trailing spaces, editor-added blank lines) do not.
"""

from __future__ import annotations

import hashlib
import json
import re

# ─────────────────────────────────────────────────────────────────────────────
# Private normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalise_system_prompt(text: str) -> str:
    """
    Normalise a system prompt string for deterministic hashing.

    The goal is to produce an identical normalised form for prompts that
    differ only in insignificant whitespace, while preserving all
    semantically meaningful content.

    Rules (applied in order)
    ────────────────────────
    1. Split the text into individual lines.
    2. Strip leading and trailing whitespace from **each** line.
       This removes editor-introduced indentation and trailing spaces
       without altering the words on the line.
    3. Rejoin the stripped lines with ``\\n``.
    4. Strip leading and trailing blank lines from the **entire** result.
       Internal blank lines (paragraph breaks) are preserved because they
       may carry structural meaning in multi-section prompts.
    5. Do **not** lowercase.  Case is semantic in prompts — "NEVER" and
       "never" may carry different emphasis for the LLM.

    Parameters
    ──────────
    text : The raw system prompt string (may contain arbitrary whitespace).

    Returns
    ───────
    str : Normalised text ready for hashing.  Empty string if the input
          is empty or whitespace-only.
    """
    # Step 1–2: split into lines, strip each line individually
    lines = [line.strip() for line in text.splitlines()]

    # Step 3–4: rejoin and strip leading/trailing blank lines
    # str.strip() on the joined result removes any leading or trailing
    # newlines that resulted from blank lines at the edges.
    normalised = "\n".join(lines).strip()

    return normalised


def _normalise_output(text: str) -> str:
    """
    Normalise LLM-generated output text for deterministic hashing.

    LLM outputs sometimes contain inconsistent spacing (double spaces,
    trailing whitespace) that is not semantically meaningful.  This
    function collapses those variations so that outputs differing only
    in spacing produce the same hash.

    Rules (applied in order)
    ────────────────────────
    1. Strip leading and trailing whitespace from the entire string.
    2. Collapse runs of two or more consecutive space characters into a
       single space.  Only ASCII space (U+0020) is targeted — newlines,
       tabs, and other whitespace are left intact so that paragraph
       structure is preserved.
    3. Preserve punctuation exactly as-is.
    4. Preserve letter casing exactly as-is.
    5. Preserve sentence and line order exactly as-is.

    Parameters
    ──────────
    text : The raw LLM-generated output string.

    Returns
    ───────
    str : Normalised text ready for hashing.  Empty string if the input
          is empty or whitespace-only.
    """
    # Step 1: remove leading/trailing whitespace
    stripped = text.strip()

    # Step 2: collapse runs of 2+ spaces to a single space
    # The regex targets only the ASCII space character (not \s which
    # would also match newlines and tabs).
    collapsed = re.sub(r" {2,}", " ", stripped)

    return collapsed


# ─────────────────────────────────────────────────────────────────────────────
# Public hash functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_payload_hash(payload_dict: dict) -> str:
    """
    Compute a stable SHA-256 hex digest for an AxisPayload dictionary.

    The dictionary is serialised with sorted keys so the hash is
    deterministic regardless of Python dict insertion order.

    This function accepts a plain ``dict`` (not a Pydantic model) so it
    can be used without importing schema types, keeping the hashing module
    dependency-free.  Callers typically pass ``payload.model_dump()``.

    Parameters
    ──────────
    payload_dict : A plain dict representation of an AxisPayload
                   (e.g. from ``AxisPayload.model_dump()``).

    Returns
    ───────
    str : 64-character lowercase hex SHA-256 digest.
    """
    # Canonical JSON: sorted keys ensure order-independence,
    # ensure_ascii=False preserves Unicode characters faithfully.
    canonical = json.dumps(payload_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_system_prompt_hash(prompt_text: str) -> str:
    """
    Compute a SHA-256 hex digest of a normalised system prompt.

    The prompt text is normalised before hashing (see
    ``_normalise_system_prompt`` for the full rule set).  This ensures
    that prompts differing only in insignificant whitespace produce
    the same digest.

    Parameters
    ──────────
    prompt_text : The raw system prompt string.

    Returns
    ───────
    str : 64-character lowercase hex SHA-256 digest.
    """
    normalised = _normalise_system_prompt(prompt_text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def compute_output_hash(output_text: str) -> str:
    """
    Compute a SHA-256 hex digest of normalised LLM output text.

    The output is normalised before hashing (see ``_normalise_output``
    for the full rule set).  This ensures that outputs differing only
    in extra spaces produce the same digest.

    Parameters
    ──────────
    output_text : The raw LLM-generated text.

    Returns
    ───────
    str : 64-character lowercase hex SHA-256 digest.
    """
    normalised = _normalise_output(output_text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def compute_ipc_id(
    *,
    input_hash: str,
    system_prompt_hash: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> str:
    """
    Compute the Interpretive Provenance Chain (IPC) identifier.

    The IPC ID is a single SHA-256 digest that uniquely fingerprints the
    complete set of variables influencing a generation.  Two generations
    with the same IPC ID used identical inputs, prompts, models, and
    sampling parameters — they should produce identical (or near-identical)
    outputs.

    Formula
    ───────
    ::

        SHA-256(
            input_hash
            + ":" + system_prompt_hash
            + ":" + model
            + ":" + str(temperature)
            + ":" + str(max_tokens)
            + ":" + str(seed)
        )

    The colon ``:`` delimiter is a non-hex character that prevents
    accidental collisions from field concatenation (e.g. hash "ab" +
    model "cd" vs hash "abc" + model "d" would collide without a
    delimiter).

    Parameters
    ──────────
    input_hash         : SHA-256 hex digest of the canonical AxisPayload.
    system_prompt_hash : SHA-256 hex digest of the normalised system prompt.
    model              : Ollama model identifier (e.g. ``"gemma2:2b"``).
    temperature        : Sampling temperature used for generation.
    max_tokens         : Token budget (Ollama ``num_predict``).
    seed               : RNG seed from the AxisPayload.

    Returns
    ───────
    str : 64-character lowercase hex SHA-256 digest (the IPC ID).
    """
    # Build the composite string from all provenance fields.
    # Each field is separated by a colon to ensure unambiguous parsing.
    parts = [
        input_hash,
        system_prompt_hash,
        model,
        str(temperature),
        str(max_tokens),
        str(seed),
    ]
    combined = ":".join(parts)

    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
