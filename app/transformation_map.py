"""
app/transformation_map.py
-----------------------------------------------------------------------------
Clause-Level Alignment Layer (Transformation Map) for the Axis Descriptor Lab.

Why a dedicated module?
-----------------------
The word-level diff (client-side LCS) is too granular — clause rewrites appear
as a long sequence of single-word insertions and deletions, obscuring the
structural change.  The signal isolation layer (``signal_isolation.py``) is
lexically useful but structure-blind (set difference, not positional).

The Transformation Map fills the gap by extracting clause-scale replacement
pairs — showing *what chunk of text was replaced by what chunk* — without
semantic interpretation.

Pipeline (sentence-aware alignment)
------------------------------------
1. **Normalise** — collapse whitespace, strip edges.
2. **Sentence split** — ``nltk.sent_tokenize()`` on both texts.
3. **Sentence alignment** — ``difflib.SequenceMatcher`` on sentence lists
   to pair sentences (equal, replace, insert, delete).
4. **Token-level alignment within matched sentence pairs** — for each
   "replace" sentence pair, run ``difflib.SequenceMatcher`` on
   ``nltk.word_tokenize()`` tokens and extract "replace" opcodes.
5. **For "equal" sentence pairs** — skip (no changes).
6. **For insert/delete-only sentences** — optionally included via the
   ``include_all`` parameter.  When False (default), only replace
   operations are shown.  When True, inserts and deletes appear as
   rows with an empty removed or added side.

Noise reduction
---------------
- Ignore replacements where both sides are a single stopword.
- Merge adjacent replace operations into a single row.
- Normalise whitespace before alignment.

NLTK data requirements
----------------------
Reuses the same NLTK data packages as ``signal_isolation.py``:
``punkt_tab``, ``stopwords``.  These are ensured at import time by
``signal_isolation._ensure_nltk_data()`` which runs before this module
is typically imported.
"""

from __future__ import annotations

import difflib
import re

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Reuse the NLTK data bootstrap from signal_isolation to ensure punkt_tab
# and stopwords are available.  Importing signal_isolation triggers the
# download check at module load time.
import app.signal_isolation  # noqa: F401 — side-effect import

# Frozen set of English stopwords for O(1) membership testing.
_ENGLISH_STOPWORDS: frozenset[str] = frozenset(stopwords.words("english"))


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def _is_single_stopword(text: str) -> bool:
    """Return True if *text* is a single English stopword (case-insensitive)."""
    stripped = text.strip().lower()
    # Must be a single token with no internal spaces after stripping
    if " " in stripped:
        return False
    return stripped in _ENGLISH_STOPWORDS


def _extract_token_changes(
    sentences_a: list[str],
    sentences_b: list[str],
    *,
    include_all: bool = False,
) -> list[dict[str, str]]:
    """
    Run token-level alignment on paired sentence groups and extract
    change spans.

    Parameters
    ----------
    sentences_a : Sentence(s) from the baseline side of a replace opcode.
    sentences_b : Sentence(s) from the current side of a replace opcode.
    include_all : When True, include insert and delete opcodes as well as
                  replacements.  When False, only replacements are returned.

    Returns
    -------
    List of {"removed": ..., "added": ...} dicts for each change opcode
    found at the token level, after noise filtering.
    """
    # Join sentence groups into single strings for token-level comparison
    text_a = " ".join(sentences_a)
    text_b = " ".join(sentences_b)

    tokens_a = word_tokenize(text_a)
    tokens_b = word_tokenize(text_b)

    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b)
    rows: list[dict[str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        if tag == "replace":
            removed = " ".join(tokens_a[i1:i2])
            added = " ".join(tokens_b[j1:j2])

            # Noise reduction: skip if both sides are a single stopword
            if _is_single_stopword(removed) and _is_single_stopword(added):
                continue

            rows.append({"removed": removed, "added": added})

        elif tag == "delete" and include_all:
            removed = " ".join(tokens_a[i1:i2])
            rows.append({"removed": removed, "added": ""})

        elif tag == "insert" and include_all:
            added = " ".join(tokens_b[j1:j2])
            rows.append({"removed": "", "added": added})

    return rows


def _merge_adjacent(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Merge adjacent replacement rows into single rows.

    Two rows are "adjacent" when they appear consecutively in the list
    (which preserves document order from the SequenceMatcher opcodes).
    Merging them produces a single row whose removed/added text is the
    concatenation separated by a space.
    """
    if not rows:
        return []

    merged: list[dict[str, str]] = [rows[0].copy()]

    for row in rows[1:]:
        # Always merge consecutive rows — they represent adjacent replace
        # opcodes from the same sentence pair, which together form a
        # single clause-level substitution.
        merged[-1] = {
            "removed": merged[-1]["removed"] + " " + row["removed"],
            "added": merged[-1]["added"] + " " + row["added"],
        }

    return merged


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def compute_transformation_map(
    baseline_text: str,
    current_text: str,
    *,
    include_all: bool = False,
) -> list[dict[str, str]]:
    """
    Extract clause-level change pairs between two texts.

    Returns a list of ``{"removed": "...", "added": "..."}`` dicts
    representing the structural changes found by sentence-aware
    alignment followed by token-level diffing within changed sentence
    groups.

    Parameters
    ----------
    baseline_text : The reference text (A).
    current_text  : The comparison text (B).
    include_all   : When True, include insert-only and delete-only
                    operations as rows (with an empty ``removed`` or
                    ``added`` side).  When False (default), only
                    replacement operations are returned.

    Returns
    -------
    list[dict[str, str]]
        Each dict has ``removed`` (text from A) and ``added`` (text from B).
        Empty list if the texts are identical.
    """
    # Step 1: normalise whitespace
    text_a = _normalise_whitespace(baseline_text)
    text_b = _normalise_whitespace(current_text)

    if not text_a or not text_b:
        return []

    # Step 2: sentence split
    sents_a = sent_tokenize(text_a)
    sents_b = sent_tokenize(text_b)

    # Step 3: sentence-level alignment
    sent_matcher = difflib.SequenceMatcher(None, sents_a, sents_b)
    all_rows: list[dict[str, str]] = []

    for tag, i1, i2, j1, j2 in sent_matcher.get_opcodes():
        if tag == "equal":
            # No changes — skip
            continue
        elif tag == "replace":
            # Step 4: token-level alignment within replaced sentence pairs
            changes = _extract_token_changes(
                sents_a[i1:i2],
                sents_b[j1:j2],
                include_all=include_all,
            )
            all_rows.extend(changes)
        elif tag == "delete" and include_all:
            # Entire sentence(s) deleted from A with no counterpart in B
            removed = " ".join(sents_a[i1:i2])
            all_rows.append({"removed": removed, "added": ""})
        elif tag == "insert" and include_all:
            # Entire sentence(s) inserted in B with no counterpart in A
            added = " ".join(sents_b[j1:j2])
            all_rows.append({"removed": "", "added": added})

    return all_rows
