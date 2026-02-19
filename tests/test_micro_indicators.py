"""
Tests for app/micro_indicators.py — structural pattern indicators.

Each test class targets a single indicator or functional area.  The tests
verify:
  1. Correctness of each heuristic classifier.
  2. Lexicon data is loaded and well-formed.
  3. Configuration (IndicatorConfig) tuning parameters.
  4. Determinism (same input always produces the same indicators).
  5. Fallback behaviour (tone reframing, lexical pivot).
  6. Edge cases (empty inputs, single words, matching token sets).
"""

from __future__ import annotations

import pytest

from app.micro_indicators import (
    ALL_INDICATORS,
    IndicatorConfig,
    _ABSTRACT_TERMS,
    _ABSTRACT_WORDS,
    _ALL_KNOWN_LEXICON,
    _CONCRETE_TERMS,
    _INTENSITY_INDEX,
    _PHYSICAL_WORDS,
    _check_abstraction_up,
    _check_compression,
    _check_consolidation,
    _check_embodiment_shift,
    _check_expansion,
    _check_fragmentation,
    _check_intensity,
    _check_lexical_pivot,
    _check_modality_shift,
    _tokenize_lower,
    classify_row,
    classify_rows,
)

# ── Lexicon data loading ──────────────────────────────────────────────────


class TestLexiconLoading:
    """Verify that lexicon data is loaded correctly at module import time."""

    def test_abstract_words_not_empty(self) -> None:
        """The embodiment lexicon must have abstract words loaded."""
        assert len(_ABSTRACT_WORDS) > 0

    def test_physical_words_not_empty(self) -> None:
        """The embodiment lexicon must have physical words loaded."""
        assert len(_PHYSICAL_WORDS) > 0

    def test_abstract_terms_not_empty(self) -> None:
        """The abstraction lexicon must have abstract terms loaded."""
        assert len(_ABSTRACT_TERMS) > 0

    def test_concrete_terms_not_empty(self) -> None:
        """The abstraction lexicon must have concrete terms loaded."""
        assert len(_CONCRETE_TERMS) > 0

    def test_intensity_index_not_empty(self) -> None:
        """The intensity index must have entries from the scales."""
        assert len(_INTENSITY_INDEX) > 0

    def test_all_abstract_words_lowercase(self) -> None:
        """All abstract words must be stored in lowercase for matching."""
        for w in _ABSTRACT_WORDS:
            assert w == w.lower(), f"Not lowercase: {w}"

    def test_all_physical_words_lowercase(self) -> None:
        """All physical words must be stored in lowercase for matching."""
        for w in _PHYSICAL_WORDS:
            assert w == w.lower(), f"Not lowercase: {w}"

    def test_all_concrete_terms_lowercase(self) -> None:
        """All concrete terms must be stored in lowercase for matching."""
        for w in _CONCRETE_TERMS:
            assert w == w.lower(), f"Not lowercase: {w}"

    def test_all_abstract_terms_lowercase(self) -> None:
        """All abstract terms must be stored in lowercase for matching."""
        for w in _ABSTRACT_TERMS:
            assert w == w.lower(), f"Not lowercase: {w}"

    def test_intensity_index_keys_lowercase(self) -> None:
        """All intensity index keys must be lowercase."""
        for key in _INTENSITY_INDEX:
            assert key == key.lower(), f"Not lowercase: {key}"

    def test_known_abstract_word_present(self) -> None:
        """Spot-check: 'tension' should be in the abstract word set."""
        assert "tension" in _ABSTRACT_WORDS

    def test_known_physical_word_present(self) -> None:
        """Spot-check: 'hands' should be in the physical word set."""
        assert "hands" in _PHYSICAL_WORDS

    def test_known_intensity_word_indexed(self) -> None:
        """Spot-check: 'uneasy' should be in the intensity index."""
        assert "uneasy" in _INTENSITY_INDEX

    def test_all_known_lexicon_is_union(self) -> None:
        """The union set must contain all individual lexicon entries."""
        assert _ABSTRACT_WORDS <= _ALL_KNOWN_LEXICON
        assert _PHYSICAL_WORDS <= _ALL_KNOWN_LEXICON
        assert _ABSTRACT_TERMS <= _ALL_KNOWN_LEXICON
        assert _CONCRETE_TERMS <= _ALL_KNOWN_LEXICON


# ── Tokenisation helper ───────────────────────────────────────────────────


class TestTokenizeLower:
    """Verify the tokenise-and-lowercase helper."""

    def test_basic_tokenisation(self) -> None:
        """Words are split and lowercased."""
        result = _tokenize_lower("The Dark Figure")
        assert result == ["the", "dark", "figure"]

    def test_punctuation_removed(self) -> None:
        """Punctuation-only tokens are discarded."""
        result = _tokenize_lower("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_empty_string(self) -> None:
        """Empty input produces empty list."""
        assert _tokenize_lower("") == []


# ── Compression ───────────────────────────────────────────────────────────


class TestCompression:
    """Verify the compression indicator heuristic."""

    def test_clear_compression(self) -> None:
        """5 tokens → 1 token should trigger compression (ratio 5:1)."""
        result = _check_compression(
            ["the", "old", "weathered", "dark", "figure"],
            ["shadow"],
            IndicatorConfig(),
        )
        assert result == "compression"

    def test_no_compression_similar_length(self) -> None:
        """Equal-length token lists should not trigger compression."""
        result = _check_compression(
            ["old", "dark"],
            ["young", "bright"],
            IndicatorConfig(),
        )
        assert result is None

    def test_min_tokens_threshold(self) -> None:
        """Compression should not fire when removed count < min_tokens."""
        result = _check_compression(
            ["old"],
            ["x"],
            IndicatorConfig(min_tokens=2),
        )
        assert result is None

    def test_custom_ratio(self) -> None:
        """A higher ratio should be harder to trigger."""
        result = _check_compression(
            ["old", "dark", "figure"],  # 3 tokens
            ["shadow"],  # 1 token → ratio 3:1
            IndicatorConfig(compression_ratio=4.0),
        )
        assert result is None  # 3 < 4*1

    def test_empty_added_no_crash(self) -> None:
        """Empty added tokens should not crash or trigger."""
        result = _check_compression(["old", "dark"], [], IndicatorConfig())
        assert result is None


# ── Expansion ─────────────────────────────────────────────────────────────


class TestExpansion:
    """Verify the expansion indicator heuristic."""

    def test_clear_expansion(self) -> None:
        """1 token → 5 tokens should trigger expansion."""
        result = _check_expansion(
            ["shadow"],
            ["the", "old", "weathered", "dark", "figure"],
            IndicatorConfig(),
        )
        assert result == "expansion"

    def test_no_expansion_similar_length(self) -> None:
        """Equal-length token lists should not trigger expansion."""
        result = _check_expansion(
            ["old", "dark"],
            ["young", "bright"],
            IndicatorConfig(),
        )
        assert result is None

    def test_empty_removed_no_crash(self) -> None:
        """Empty removed tokens should not crash or trigger."""
        result = _check_expansion([], ["a", "b", "c"], IndicatorConfig())
        assert result is None


# ── Embodiment shift ──────────────────────────────────────────────────────


class TestEmbodimentShift:
    """Verify the embodiment shift indicator (abstract → physical)."""

    def test_abstract_to_physical(self) -> None:
        """Abstract words removed + physical words added → embodiment shift."""
        result = _check_embodiment_shift(["tension", "burden"], ["hands", "face"])
        assert result == "embodiment shift"

    def test_no_shift_same_domain(self) -> None:
        """Physical-to-physical swap should not trigger."""
        result = _check_embodiment_shift(["hands", "face"], ["eyes", "posture"])
        assert result is None

    def test_no_shift_abstract_only(self) -> None:
        """Abstract-to-abstract swap should not trigger."""
        result = _check_embodiment_shift(["tension"], ["burden"])
        assert result is None

    def test_partial_overlap(self) -> None:
        """At least one abstract removed + one physical added is enough."""
        result = _check_embodiment_shift(["tension", "goblin"], ["hands", "goblin"])
        assert result == "embodiment shift"


# ── Abstraction ↑ ─────────────────────────────────────────────────────────


class TestAbstractionUp:
    """Verify the abstraction increase indicator (concrete → abstract)."""

    def test_concrete_to_abstract(self) -> None:
        """Concrete removed + abstract added → abstraction ↑."""
        result = _check_abstraction_up(["coat", "boots"], ["authority", "influence"])
        assert result == "abstraction \u2191"

    def test_no_shift_abstract_to_abstract(self) -> None:
        """Abstract-to-abstract should not trigger."""
        result = _check_abstraction_up(["authority"], ["influence"])
        assert result is None

    def test_no_shift_concrete_to_concrete(self) -> None:
        """Concrete-to-concrete should not trigger."""
        result = _check_abstraction_up(["coat"], ["boots"])
        assert result is None


# ── Intensity ─────────────────────────────────────────────────────────────


class TestIntensity:
    """Verify the intensity shift indicator (up/down on known scales)."""

    def test_intensity_increase(self) -> None:
        """Moving from lower to higher on a scale → intensity ↑."""
        result = _check_intensity(["uneasy"], ["perilous"])
        assert result == "intensity \u2191"

    def test_intensity_decrease(self) -> None:
        """Moving from higher to lower on a scale → intensity ↓."""
        result = _check_intensity(["perilous"], ["uneasy"])
        assert result == "intensity \u2193"

    def test_same_word_no_change(self) -> None:
        """Same word on both sides → no intensity shift."""
        result = _check_intensity(["uneasy"], ["uneasy"])
        assert result is None

    def test_no_scale_words(self) -> None:
        """Words not on any scale → no intensity shift."""
        result = _check_intensity(["goblin"], ["shadow"])
        assert result is None

    def test_different_scales_no_match(self) -> None:
        """Words from different scales should not trigger."""
        # "uneasy" is on unease_scale, "fragile" is on strength_scale
        result = _check_intensity(["uneasy"], ["fragile"])
        assert result is None


# ── Consolidation ─────────────────────────────────────────────────────────


class TestConsolidation:
    """Verify the consolidation indicator (sentence count decreases)."""

    def test_sentence_decrease(self) -> None:
        """Multiple sentences → fewer sentences → consolidation."""
        result = _check_consolidation(
            "The goblin stands. It watches. Waiting.",
            "The goblin stands, watching and waiting.",
        )
        assert result == "consolidation"

    def test_no_consolidation_single_sentence(self) -> None:
        """Single sentence on both sides → no consolidation."""
        result = _check_consolidation("The goblin stands.", "The goblin waits.")
        assert result is None

    def test_no_consolidation_same_count(self) -> None:
        """Same sentence count → no consolidation."""
        result = _check_consolidation(
            "The goblin stands. It watches.",
            "The goblin waits. It looks.",
        )
        assert result is None


# ── Fragmentation ─────────────────────────────────────────────────────────


class TestFragmentation:
    """Verify the fragmentation indicator (sentence count increases)."""

    def test_sentence_increase(self) -> None:
        """Fewer sentences → more sentences → fragmentation."""
        result = _check_fragmentation(
            "The goblin stands watching.",
            "The goblin stands. It watches.",
        )
        assert result == "fragmentation"

    def test_no_fragmentation_single_to_single(self) -> None:
        """Single sentence on both sides → no fragmentation."""
        result = _check_fragmentation("The goblin stands.", "The goblin waits.")
        assert result is None


# ── Modality shift ────────────────────────────────────────────────────────


class TestModalityShift:
    """Verify the modality shift indicator (verb/adjective density change)."""

    def test_returns_string_or_none(self) -> None:
        """Result must be either 'modality shift' or None."""
        result = _check_modality_shift(
            ["running", "bright", "fast"],
            ["the", "upon", "table"],
            IndicatorConfig(modality_density_threshold=0.3),
        )
        assert result is None or result == "modality shift"

    def test_empty_tokens_no_crash(self) -> None:
        """Empty token lists should not crash."""
        result = _check_modality_shift([], [], IndicatorConfig())
        assert result is None

    def test_single_token_no_crash(self) -> None:
        """Single tokens should not crash."""
        result = _check_modality_shift(["bright"], ["table"], IndicatorConfig())
        assert result is None or result == "modality shift"


# ── Lexical pivot ─────────────────────────────────────────────────────────


class TestLexicalPivot:
    """Verify the lexical pivot indicator (rare content word swap)."""

    def test_rare_word_swap(self) -> None:
        """Two rare words (not in any lexicon, not stopwords) → pivot."""
        result = _check_lexical_pivot(["threshold"], ["precipice"])
        assert result == "lexical pivot"

    def test_no_pivot_known_words(self) -> None:
        """Words in the lexicon sets are not considered 'rare'."""
        result = _check_lexical_pivot(["tension"], ["authority"])
        assert result is None

    def test_no_pivot_stopwords(self) -> None:
        """Stopword-only tokens should not trigger."""
        result = _check_lexical_pivot(["the"], ["a"])
        assert result is None

    def test_no_pivot_empty(self) -> None:
        """Empty token lists should not trigger."""
        result = _check_lexical_pivot([], [])
        assert result is None


# ── classify_row (public API) ─────────────────────────────────────────────


class TestClassifyRow:
    """Verify the main classification function."""

    def test_returns_list(self) -> None:
        """Result must always be a list."""
        result = classify_row("the old dark weathered figure", "shadow")
        assert isinstance(result, list)

    def test_empty_inputs_no_indicators(self) -> None:
        """Both sides empty → empty list."""
        result = classify_row("", "")
        assert result == []

    def test_whitespace_only_no_indicators(self) -> None:
        """Whitespace-only inputs → empty list."""
        result = classify_row("   ", "   ")
        assert result == []

    def test_compression_detected(self) -> None:
        """Clear compression case: many tokens → single token."""
        result = classify_row(
            "etched with lines that speak of hardship",
            "suggesting",
        )
        assert "compression" in result

    def test_expansion_detected(self) -> None:
        """Clear expansion case: single token → many tokens."""
        result = classify_row(
            "burden",
            "a heavy burden weighing on him",
        )
        assert "expansion" in result

    def test_embodiment_shift_detected(self) -> None:
        """Abstract→physical lexicon match."""
        result = classify_row("tension hangs", "asymmetry")
        assert "embodiment shift" in result

    def test_intensity_increase_detected(self) -> None:
        """Lower→higher on a known scale."""
        result = classify_row("uneasy atmosphere", "perilous atmosphere")
        assert "intensity \u2191" in result

    def test_multiple_indicators_possible(self) -> None:
        """A row can have more than one indicator."""
        # This should trigger compression (many→few) + possibly embodiment
        result = classify_row(
            "tension and burden and uncertainty fill the room",
            "hands tremble",
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_enabled_filter(self) -> None:
        """When enabled is set, only those indicators should appear."""
        config = IndicatorConfig(enabled=("compression",))
        result = classify_row(
            "the old dark weathered figure stands near the gate",
            "shadow",
            config=config,
        )
        for ind in result:
            assert ind == "compression"

    def test_determinism(self) -> None:
        """Same inputs must always produce the same indicators."""
        r1 = classify_row("the dark figure lurks near", "a bright shadow waits")
        r2 = classify_row("the dark figure lurks near", "a bright shadow waits")
        assert r1 == r2

    def test_fallback_fires_when_no_structural(self) -> None:
        """When no structural indicator matches, a fallback should fire."""
        # Two different words, similar length, not in any lexicon
        result = classify_row("pleasant", "agreeable")
        assert len(result) >= 1
        # Should be one of the fallback indicators
        assert any(ind in ("tone reframing", "lexical pivot", "modality shift") for ind in result)


# ── classify_rows (batch API) ────────────────────────────────────────────


class TestClassifyRows:
    """Verify the batch classification function."""

    def test_matches_row_count(self) -> None:
        """Output length must match input length."""
        rows = [
            {"removed": "old", "added": "young"},
            {"removed": "dark", "added": "bright"},
        ]
        result = classify_rows(rows)
        assert len(result) == len(rows)

    def test_empty_rows(self) -> None:
        """Empty input produces empty output."""
        assert classify_rows([]) == []

    def test_each_element_is_list(self) -> None:
        """Each element in the result must be a list of strings."""
        rows = [{"removed": "old figure", "added": "young goblin"}]
        result = classify_rows(rows)
        assert isinstance(result[0], list)
        for ind in result[0]:
            assert isinstance(ind, str)


# ── IndicatorConfig ───────────────────────────────────────────────────────


class TestIndicatorConfig:
    """Verify configuration defaults and custom values."""

    def test_defaults(self) -> None:
        """Default config uses conservative values."""
        config = IndicatorConfig()
        assert config.compression_ratio == 2.0
        assert config.expansion_ratio == 2.0
        assert config.min_tokens == 2
        assert config.modality_density_threshold == 0.3
        assert config.enabled is None

    def test_custom_values(self) -> None:
        """Custom values are applied correctly."""
        config = IndicatorConfig(
            compression_ratio=3.0,
            expansion_ratio=4.0,
            min_tokens=3,
            modality_density_threshold=0.5,
            enabled=("compression", "expansion"),
        )
        assert config.compression_ratio == 3.0
        assert config.expansion_ratio == 4.0
        assert config.min_tokens == 3
        assert config.enabled == ("compression", "expansion")

    def test_frozen(self) -> None:
        """IndicatorConfig is immutable (frozen dataclass)."""
        config = IndicatorConfig()
        with pytest.raises(AttributeError):
            config.compression_ratio = 5.0  # type: ignore[misc]


# ── ALL_INDICATORS constant ──────────────────────────────────────────────


class TestClassifyRowIntegrationPaths:
    """Cover classify_row() integration paths for consolidation, fragmentation,
    abstraction ↑, and lexical pivot that only fire inside the main dispatcher."""

    def test_consolidation_via_classify_row(self) -> None:
        """Sentence count decrease should surface 'consolidation' through classify_row."""
        result = classify_row(
            "The goblin stands. It watches. Waiting.",
            "The goblin stands and watches.",
        )
        assert "consolidation" in result

    def test_fragmentation_via_classify_row(self) -> None:
        """Sentence count increase should surface 'fragmentation' through classify_row."""
        result = classify_row(
            "The goblin stands and watches.",
            "The goblin stands. It watches. Waiting.",
        )
        assert "fragmentation" in result

    def test_abstraction_up_via_classify_row(self) -> None:
        """Concrete→abstract lexicon match through classify_row."""
        # "face" is in _CONCRETE_TERMS, "burden" is in _ABSTRACT_TERMS
        result = classify_row("face", "burden")
        assert "abstraction \u2191" in result

    def test_lexical_pivot_via_classify_row(self) -> None:
        """Rare content-word swap when tone reframing is disabled → lexical pivot."""
        # Disable tone reframing so the lexical pivot fallback can fire
        config = IndicatorConfig(enabled=("lexical pivot",))
        result = classify_row("crevice", "fracture", config=config)
        assert "lexical pivot" in result

    def test_intensity_skips_unmatched_added_word(self) -> None:
        """When removed word is on a scale but added word is not, skip gracefully."""
        # "uneasy" is on unease_scale; "goblin" is not on any scale
        result = _check_intensity(["uneasy"], ["goblin"])
        assert result is None


class TestNltkFallback:
    """Cover the NLTK download-failure and pos_tag exception paths."""

    def test_pos_tag_exception_returns_none(self) -> None:
        """If nltk.pos_tag raises, _check_modality_shift should return None."""
        import unittest.mock

        with unittest.mock.patch("app.micro_indicators.nltk.pos_tag", side_effect=RuntimeError):
            result = _check_modality_shift(["old", "dark"], ["new", "bright"], IndicatorConfig())
            assert result is None

    def test_nltk_download_failure_logged(self) -> None:
        """If NLTK download fails, the warning path should execute without raising."""
        import unittest.mock

        from app.micro_indicators import _ensure_pos_tagger_data

        with (
            unittest.mock.patch("app.micro_indicators.nltk.data.find", side_effect=LookupError),
            unittest.mock.patch(
                "app.micro_indicators.nltk.download", side_effect=OSError("network error")
            ),
            unittest.mock.patch("app.micro_indicators.logger") as mock_logger,
        ):
            _ensure_pos_tagger_data()
            assert mock_logger.warning.called


class TestAllIndicators:
    """Verify the canonical indicator list."""

    def test_has_expected_count(self) -> None:
        """There should be 11 indicator names (including both intensity directions)."""
        assert len(ALL_INDICATORS) == 11

    def test_contains_known_indicators(self) -> None:
        """Spot-check: known indicator names must be present."""
        assert "compression" in ALL_INDICATORS
        assert "expansion" in ALL_INDICATORS
        assert "embodiment shift" in ALL_INDICATORS
        assert "abstraction \u2191" in ALL_INDICATORS
        assert "intensity \u2191" in ALL_INDICATORS
        assert "intensity \u2193" in ALL_INDICATORS
        assert "consolidation" in ALL_INDICATORS
        assert "fragmentation" in ALL_INDICATORS
        assert "tone reframing" in ALL_INDICATORS
        assert "modality shift" in ALL_INDICATORS
        assert "lexical pivot" in ALL_INDICATORS

    def test_all_strings(self) -> None:
        """Every entry must be a non-empty string."""
        for name in ALL_INDICATORS:
            assert isinstance(name, str)
            assert len(name.strip()) > 0
