"""
tests/test_transformation_map.py
─────────────────────────────────────────────────────────────────────────────
Tests for app/transformation_map.py — clause-level alignment layer.

Each test class targets a specific aspect of the transformation map pipeline.
The tests verify:

1. Basic replacement extraction (clause-level pairs).
2. Noise reduction (stopword-only replacements filtered out).
3. Adjacent merging behaviour.
4. Edge cases (empty strings, identical texts, insert/delete-only changes).
5. Determinism (same input always produces same output).
"""

from __future__ import annotations

from app.transformation_map import (
    _is_single_stopword,
    _merge_adjacent,
    _normalise_whitespace,
    compute_transformation_map,
)

# ─────────────────────────────────────────────────────────────────────────────
# _normalise_whitespace
# ─────────────────────────────────────────────────────────────────────────────


class TestNormaliseWhitespace:
    """Verify whitespace normalisation helper."""

    def test_collapses_multiple_spaces(self) -> None:
        assert _normalise_whitespace("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines(self) -> None:
        assert _normalise_whitespace("hello\t\nworld") == "hello world"

    def test_strips_edges(self) -> None:
        assert _normalise_whitespace("  hello world  ") == "hello world"

    def test_empty_string(self) -> None:
        assert _normalise_whitespace("") == ""

    def test_whitespace_only(self) -> None:
        assert _normalise_whitespace("   \t\n  ") == ""


# ─────────────────────────────────────────────────────────────────────────────
# _is_single_stopword
# ─────────────────────────────────────────────────────────────────────────────


class TestIsSingleStopword:
    """Verify single-stopword detection."""

    def test_common_stopwords(self) -> None:
        assert _is_single_stopword("the") is True
        assert _is_single_stopword("a") is True
        assert _is_single_stopword("is") is True

    def test_case_insensitive(self) -> None:
        assert _is_single_stopword("The") is True
        assert _is_single_stopword("THE") is True

    def test_content_word_not_stopword(self) -> None:
        assert _is_single_stopword("goblin") is False
        assert _is_single_stopword("weathered") is False

    def test_multi_word_not_single_stopword(self) -> None:
        assert _is_single_stopword("the a") is False

    def test_empty_string(self) -> None:
        assert _is_single_stopword("") is False


# ─────────────────────────────────────────────────────────────────────────────
# _merge_adjacent
# ─────────────────────────────────────────────────────────────────────────────


class TestMergeAdjacent:
    """Verify adjacent row merging."""

    def test_empty_list(self) -> None:
        assert _merge_adjacent([]) == []

    def test_single_row_unchanged(self) -> None:
        rows = [{"removed": "old text", "added": "new text"}]
        result = _merge_adjacent(rows)
        assert len(result) == 1
        assert result[0] == {"removed": "old text", "added": "new text"}

    def test_two_rows_merged(self) -> None:
        rows = [
            {"removed": "dark", "added": "bright"},
            {"removed": "figure", "added": "shadow"},
        ]
        result = _merge_adjacent(rows)
        assert len(result) == 1
        assert result[0]["removed"] == "dark figure"
        assert result[0]["added"] == "bright shadow"

    def test_adjacent_tokens_stay_grouped(self) -> None:
        """Adjacent changed tokens form a single replace opcode and thus
        a single row — SequenceMatcher groups them naturally."""
        baseline = "The old dark goblin stands near the gate."
        current = "The young bright goblin stands near the gate."
        result = compute_transformation_map(baseline, current)
        # "old dark" → "young bright" is one contiguous replacement
        assert len(result) == 1
        assert "old" in result[0]["removed"]
        assert "young" in result[0]["added"]


# ─────────────────────────────────────────────────────────────────────────────
# compute_transformation_map
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeTransformationMap:
    """Integration tests for the complete transformation map pipeline."""

    def test_identical_texts_no_rows(self) -> None:
        """Identical texts should produce no replacement rows."""
        text = "The old goblin stands near the gate."
        result = compute_transformation_map(text, text)
        assert result == []

    def test_empty_baseline_no_rows(self) -> None:
        result = compute_transformation_map("", "Some text here.")
        assert result == []

    def test_empty_current_no_rows(self) -> None:
        result = compute_transformation_map("Some text here.", "")
        assert result == []

    def test_basic_word_replacement(self) -> None:
        """A single word change in a sentence should produce a replacement row."""
        baseline = "The old goblin stands near the gate."
        current = "The young goblin stands near the gate."
        result = compute_transformation_map(baseline, current)
        assert len(result) >= 1
        # At least one row should contain "old" removed and "young" added
        found = any("old" in r["removed"] and "young" in r["added"] for r in result)
        assert found, f"Expected old→young replacement in {result}"

    def test_multi_word_replacement(self) -> None:
        """Multiple word changes should be captured."""
        baseline = "A dark weathered figure lurks by the broken wall."
        current = "A bright youthful figure waits by the stone wall."
        result = compute_transformation_map(baseline, current)
        assert len(result) >= 1

    def test_stopword_only_replacement_filtered(self) -> None:
        """Noise reduction: when both sides of a replacement are a single
        stopword (e.g. "The" → "A"), the row should be silently discarded.

        This prevents trivial article/preposition swaps from cluttering the
        transformation map with non-meaningful changes.  Content-word
        replacements (e.g. "goblin" → "creature") must still be kept.
        """
        # "The cat sat." vs "A cat sat." — only difference is "The" → "A",
        # both of which are NLTK English stopwords.
        baseline = "The cat sat."
        current = "A cat sat."
        result = compute_transformation_map(baseline, current)
        for row in result:
            assert not (
                row["removed"].strip().lower() == "the" and row["added"].strip().lower() == "a"
            ), f"Stopword-only replacement not filtered: {row}"

    def test_determinism(self) -> None:
        """Same inputs must always produce the same output."""
        baseline = "The figure stands in shadow, clutching a worn staff."
        current = "The creature waits in darkness, holding a polished blade."
        result1 = compute_transformation_map(baseline, current)
        result2 = compute_transformation_map(baseline, current)
        assert result1 == result2

    def test_whitespace_normalisation(self) -> None:
        """Extra whitespace should not affect results."""
        baseline = "The   old   goblin  stands."
        current = "The young goblin stands."
        result_messy = compute_transformation_map(baseline, current)
        result_clean = compute_transformation_map(
            "The old goblin stands.",
            "The young goblin stands.",
        )
        assert result_messy == result_clean

    def test_multi_sentence_replacement(self) -> None:
        """Replacements across multiple sentences should be captured."""
        baseline = "The goblin is old. It stands near the gate."
        current = "The goblin is young. It waits by the door."
        result = compute_transformation_map(baseline, current)
        assert len(result) >= 1

    def test_returns_list_of_dicts(self) -> None:
        """Each row should be a dict with 'removed' and 'added' keys."""
        baseline = "Dark clouds gather overhead."
        current = "Bright stars shine overhead."
        result = compute_transformation_map(baseline, current)
        for row in result:
            assert isinstance(row, dict)
            assert "removed" in row
            assert "added" in row
            assert isinstance(row["removed"], str)
            assert isinstance(row["added"], str)

    def test_include_all_false_excludes_inserts_deletes(self) -> None:
        """Default mode (include_all=False) returns replacement-only rows.

        Pure insertions (text added in B with no counterpart in A) and pure
        deletions (text removed from A with no replacement in B) are excluded.
        This matches the "Replacements only" UI toggle state and keeps the
        table focused on clause-level substitutions.
        """
        baseline = "The goblin stands."
        current = "The goblin stands. A new sentence appears."
        result = compute_transformation_map(baseline, current, include_all=False)
        for row in result:
            assert "new sentence" not in row["added"]

    def test_include_all_true_includes_inserts(self) -> None:
        """include_all=True should include inserted sentences."""
        baseline = "The goblin stands."
        current = "The goblin stands. A new sentence appears."
        result = compute_transformation_map(baseline, current, include_all=True)
        # The inserted sentence should appear with empty removed
        found = any(row["removed"] == "" and "new sentence" in row["added"] for row in result)
        assert found, f"Expected insert row in {result}"

    def test_include_all_true_includes_deletes(self) -> None:
        """include_all=True should surface pure deletions as rows.

        When text from A is removed entirely (no replacement in B), the row
        should have the deleted text in "removed" and an empty string in
        "added".  This corresponds to the "All changes" UI toggle state,
        where the added column shows an em dash for the empty side.
        """
        baseline = "The goblin stands. An old sentence exists."
        current = "The goblin stands."
        result = compute_transformation_map(baseline, current, include_all=True)
        found = any(row["added"] == "" and "old sentence" in row["removed"] for row in result)
        assert found, f"Expected delete row in {result}"
