"""
tests/test_signal_isolation.py
─────────────────────────────────────────────────────────────────────────────
Tests for app/signal_isolation.py — NLP pipeline for content-word delta.

Each test class targets a single function.  The tests verify:

1. Correctness of each pipeline step (tokenise, lemmatise, filter, extract).
2. Determinism (same input always produces same output).
3. Edge cases (empty strings, whitespace-only, all stopwords).
4. The complete ``compute_delta()`` function (integration of all steps).
"""

from __future__ import annotations

from app.signal_isolation import (
    _filter_stopwords,
    _lemmatise,
    _tokenise,
    compute_delta,
    extract_content_lemmas,
)

# ─────────────────────────────────────────────────────────────────────────────
# _tokenise
# ─────────────────────────────────────────────────────────────────────────────


class TestTokenise:
    """Verify tokenisation: lowercasing, alpha filtering, punctuation removal."""

    def test_basic_sentence(self) -> None:
        """A simple sentence should produce lowercase word tokens."""
        result = _tokenise("The cat sat on the mat.")
        assert "the" in result
        assert "cat" in result
        assert "sat" in result
        assert "mat" in result

    def test_lowercases_all_tokens(self) -> None:
        """All tokens must be lowercased regardless of input casing."""
        result = _tokenise("HELLO World")
        assert result == ["hello", "world"]

    def test_filters_pure_punctuation(self) -> None:
        """Punctuation-only tokens (commas, periods, dashes) must be excluded."""
        result = _tokenise("Hello, world! Yes -- no.")
        # Only alphabetic tokens should survive
        alpha_only = [t for t in result if t.isalpha()]
        assert result == alpha_only

    def test_filters_pure_numbers(self) -> None:
        """Pure numeric tokens must be excluded from the result."""
        result = _tokenise("There are 42 goblins and 7 trolls.")
        assert "42" not in result
        assert "7" not in result
        assert "goblins" in result
        assert "trolls" in result

    def test_empty_string_returns_empty_list(self) -> None:
        """Empty input produces an empty list."""
        assert _tokenise("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        """Whitespace-only input produces an empty list."""
        assert _tokenise("   \n\t  ") == []

    def test_deterministic(self) -> None:
        """Same input must always produce the same output."""
        text = "The weathered figure carries a burden."
        assert _tokenise(text) == _tokenise(text)

    def test_preserves_contractions(self) -> None:
        """Contractions should be split but alpha parts preserved."""
        result = _tokenise("He can't do it.")
        # NLTK splits "can't" into "ca" and "n't"
        assert any("ca" in t or "n't" in t for t in result)


# ─────────────────────────────────────────────────────────────────────────────
# _lemmatise
# ─────────────────────────────────────────────────────────────────────────────


class TestLemmatise:
    """Verify lemmatisation: verb and noun reduction to base forms."""

    def test_verb_inflections_reduced(self) -> None:
        """Common verb inflections should reduce to their base form."""
        result = _lemmatise(["carries", "failing", "walked"])
        assert "carry" in result
        assert "fail" in result
        assert "walk" in result

    def test_noun_plurals_reduced(self) -> None:
        """Plural nouns should reduce to singular form."""
        result = _lemmatise(["figures", "goblins", "axes"])
        assert "figure" in result
        assert "goblin" in result

    def test_already_base_form_unchanged(self) -> None:
        """Words already in base form should pass through unchanged."""
        result = _lemmatise(["walk", "carry", "figure"])
        assert result == ["walk", "carry", "figure"]

    def test_preserves_token_order(self) -> None:
        """Lemmatisation must preserve the original token order."""
        tokens = ["carries", "heavy", "burdens"]
        result = _lemmatise(tokens)
        assert len(result) == 3
        # The order should be: carry, heavy, burden
        assert result[0] == "carry"
        assert result[2] == "burden"

    def test_empty_list_returns_empty(self) -> None:
        """Empty input list produces empty output list."""
        assert _lemmatise([]) == []

    def test_deterministic(self) -> None:
        """Same input must always produce the same output."""
        tokens = ["weathered", "figures", "carrying", "burdens"]
        assert _lemmatise(tokens) == _lemmatise(tokens)

    def test_same_length_as_input(self) -> None:
        """Output list must always be the same length as input list."""
        tokens = ["running", "goblins", "dark", "threshold"]
        result = _lemmatise(tokens)
        assert len(result) == len(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# _filter_stopwords
# ─────────────────────────────────────────────────────────────────────────────


class TestFilterStopwords:
    """Verify stopword removal: function words removed, content words kept."""

    def test_removes_common_stopwords(self) -> None:
        """Articles, pronouns, and auxiliaries must be removed."""
        tokens = ["the", "cat", "is", "on", "a", "mat"]
        result = _filter_stopwords(tokens)
        assert "the" not in result
        assert "is" not in result
        assert "on" not in result
        assert "a" not in result
        # Content words survive
        assert "cat" in result
        assert "mat" in result

    def test_preserves_content_words(self) -> None:
        """Non-stopword tokens must be preserved in order."""
        tokens = ["weathered", "figure", "threshold"]
        result = _filter_stopwords(tokens)
        assert result == ["weathered", "figure", "threshold"]

    def test_all_stopwords_returns_empty(self) -> None:
        """Input consisting entirely of stopwords produces an empty list."""
        tokens = ["the", "a", "is", "are", "was", "on", "in", "at"]
        result = _filter_stopwords(tokens)
        assert result == []

    def test_empty_list_returns_empty(self) -> None:
        """Empty input produces empty output."""
        assert _filter_stopwords([]) == []

    def test_preserves_order(self) -> None:
        """Non-stopword tokens must remain in their original order."""
        tokens = ["the", "dark", "a", "figure", "and", "threshold"]
        result = _filter_stopwords(tokens)
        assert result == ["dark", "figure", "threshold"]


# ─────────────────────────────────────────────────────────────────────────────
# extract_content_lemmas
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractContentLemmas:
    """Verify the complete pipeline: tokenise + lemmatise + filter → set."""

    def test_basic_sentence_extracts_content_words(self) -> None:
        """A sentence should produce a set of content lemmas only."""
        result = extract_content_lemmas("The weathered figure carries a heavy burden.")
        assert isinstance(result, set)
        assert "figure" in result
        assert "carry" in result
        assert "heavy" in result
        assert "burden" in result
        # Stopwords must be absent
        assert "the" not in result
        assert "a" not in result

    def test_empty_string_returns_empty_set(self) -> None:
        """Empty input produces an empty set."""
        assert extract_content_lemmas("") == set()

    def test_whitespace_only_returns_empty_set(self) -> None:
        """Whitespace-only input produces an empty set."""
        assert extract_content_lemmas("   \n  ") == set()

    def test_only_stopwords_returns_empty_set(self) -> None:
        """Text consisting entirely of stopwords produces an empty set."""
        result = extract_content_lemmas("the a is are was on in at and or")
        assert result == set()

    def test_deduplicates_repeated_words(self) -> None:
        """Repeated words should appear only once in the set."""
        result = extract_content_lemmas("figure figure figure")
        assert "figure" in result
        assert len(result) == 1

    def test_deterministic(self) -> None:
        """Same input must always produce the same output."""
        text = "A weathered goblin stands near the crumbling threshold."
        assert extract_content_lemmas(text) == extract_content_lemmas(text)

    def test_case_insensitive(self) -> None:
        """Uppercase and lowercase versions of the same word should merge."""
        result = extract_content_lemmas("Figure FIGURE figure")
        assert "figure" in result
        assert len(result) == 1

    def test_inflections_merge_to_same_lemma(self) -> None:
        """Different inflections of the same word should produce one lemma."""
        result = extract_content_lemmas("carry carries carried carrying")
        assert "carry" in result


# ─────────────────────────────────────────────────────────────────────────────
# compute_delta
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeDelta:
    """Verify the content-word delta computation between two texts."""

    def test_identical_texts_produce_empty_delta(self) -> None:
        """When both texts are identical, both removed and added must be empty."""
        text = "A weathered figure stands near the threshold."
        removed, added = compute_delta(text, text)
        assert removed == []
        assert added == []

    def test_word_added_in_current(self) -> None:
        """A content word present only in B should appear in 'added'."""
        baseline = "The figure stands."
        current = "The dark figure stands menacingly."
        removed, added = compute_delta(baseline, current)
        assert "dark" in added

    def test_word_removed_from_baseline(self) -> None:
        """A content word present only in A should appear in 'removed'."""
        baseline = "The dark figure stands menacingly."
        current = "The figure stands."
        removed, added = compute_delta(baseline, current)
        assert "dark" in removed

    def test_results_alphabetically_sorted(self) -> None:
        """Both removed and added lists must be alphabetically sorted."""
        baseline = "The cat and dog walk slowly."
        current = "The bird and fish swim quickly."
        removed, added = compute_delta(baseline, current)
        assert removed == sorted(removed)
        assert added == sorted(added)

    def test_stopwords_ignored_in_delta(self) -> None:
        """Changes in stopwords should not appear in the delta."""
        baseline = "A figure stands on the mat."
        current = "The figure stands in a room."
        removed, added = compute_delta(baseline, current)
        # Stopwords must never appear in either list
        all_words = removed + added
        for stopword in ("a", "the", "on", "in"):
            assert stopword not in all_words

    def test_inflection_normalised_across_texts(self) -> None:
        """Inflectional variants should not show as differences."""
        baseline = "The figure carries a burden."
        current = "The figure carry a burden."
        removed, added = compute_delta(baseline, current)
        # Both "carries" and "carry" lemmatise to "carry"
        assert "carry" not in removed
        assert "carry" not in added

    def test_empty_baseline_all_current_words_added(self) -> None:
        """Empty baseline: all content words from current appear as added."""
        removed, added = compute_delta("", "Dark figure stands.")
        assert removed == []
        assert len(added) > 0

    def test_empty_current_all_baseline_words_removed(self) -> None:
        """Empty current: all content words from baseline appear as removed."""
        removed, added = compute_delta("Dark figure stands.", "")
        assert len(removed) > 0
        assert added == []

    def test_deterministic_across_calls(self) -> None:
        """Same inputs must always produce the same outputs."""
        a = "The weathered figure stands near the threshold."
        b = "A dark figure lurks beyond the crumbling gate."
        r1, a1 = compute_delta(a, b)
        r2, a2 = compute_delta(a, b)
        assert r1 == r2
        assert a1 == a2

    def test_symmetric_difference_swaps_on_input_swap(self) -> None:
        """Swapping A and B should swap removed and added lists."""
        a = "The dark figure stands."
        b = "The bright goblin lurks."
        removed_ab, added_ab = compute_delta(a, b)
        removed_ba, added_ba = compute_delta(b, a)
        assert removed_ab == added_ba
        assert added_ab == removed_ba

    def test_shared_words_absent_from_both_lists(self) -> None:
        """Words present in both texts should not appear in either list."""
        baseline = "The dark figure stands near the gate."
        current = "The bright figure lurks near the gate."
        removed, added = compute_delta(baseline, current)
        # "figure", "near", "gate" are shared
        for shared in ("figure", "near", "gate"):
            assert shared not in removed
            assert shared not in added
