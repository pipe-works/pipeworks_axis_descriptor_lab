"""
Tests for app/output_validator.py — 7-step OOC→IC output validation pipeline.

Each test class corresponds to one step in the pipeline.  The final class
covers integration / happy-path scenarios and logging side-effects.

The two operating modes are tested throughout:
  - strict_mode=True  (default): any violation → None
  - strict_mode=False (lenient):  recoverable violations are fixed up
"""

from __future__ import annotations

import logging

import pytest

from app.output_validator import PASSTHROUGH_SENTINEL, OutputValidator

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def strict() -> OutputValidator:
    """Strict-mode validator with the production default of 280 chars."""
    return OutputValidator(strict_mode=True, max_output_chars=280)


@pytest.fixture()
def lenient() -> OutputValidator:
    """Lenient-mode validator with the production default of 280 chars."""
    return OutputValidator(strict_mode=False, max_output_chars=280)


# ── Step 1: Empty check ───────────────────────────────────────────────────────


class TestEmptyCheck:
    """Step 1 rejects blank input before any other processing."""

    def test_empty_string_returns_none(self, strict: OutputValidator) -> None:
        assert strict.validate("") is None

    def test_whitespace_only_returns_none(self, strict: OutputValidator) -> None:
        assert strict.validate("   \t\n  ") is None

    def test_single_newline_returns_none(self, strict: OutputValidator) -> None:
        assert strict.validate("\n") is None

    def test_single_space_returns_none(self, lenient: OutputValidator) -> None:
        """Lenient mode still rejects blank input at step 1."""
        assert lenient.validate("  ") is None


# ── Step 2: PASSTHROUGH sentinel ─────────────────────────────────────────────


class TestPassthroughSentinel:
    """Step 2 treats any text starting with PASSTHROUGH as a deliberate signal."""

    def test_exact_sentinel_returns_none(self, strict: OutputValidator) -> None:
        assert strict.validate(PASSTHROUGH_SENTINEL) is None

    def test_sentinel_lowercase_returns_none(self, strict: OutputValidator) -> None:
        """Case-insensitive matching via .upper()."""
        assert strict.validate("passthrough") is None

    def test_sentinel_mixed_case_returns_none(self, strict: OutputValidator) -> None:
        assert strict.validate("Passthrough") is None

    def test_sentinel_with_trailing_text_returns_none(self, strict: OutputValidator) -> None:
        """Sentinel followed by explanation text is still rejected."""
        assert strict.validate("PASSTHROUGH: cannot translate this command") is None

    def test_partial_prefix_passes(self, strict: OutputValidator) -> None:
        """'PASS' alone does not trigger the sentinel."""
        result = strict.validate("PASS")
        assert result is not None
        assert result == "PASS"

    def test_sentinel_with_leading_whitespace_passes(self, strict: OutputValidator) -> None:
        """Leading whitespace is stripped before sentinel check; sentinel still caught."""
        assert strict.validate("  PASSTHROUGH  ") is None

    def test_lenient_mode_also_rejects_sentinel(self, lenient: OutputValidator) -> None:
        """Lenient mode does not change PASSTHROUGH handling — always rejects."""
        assert lenient.validate("passthrough more text") is None


# ── Step 3: Multi-line check ──────────────────────────────────────────────────


class TestMultiLineCheck:
    """Step 3 enforces single-line output (different behaviour by mode)."""

    def test_strict_multi_line_returns_none(self, strict: OutputValidator) -> None:
        assert strict.validate("Line one.\nLine two.") is None

    def test_strict_multi_line_logs_warning(
        self, strict: OutputValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.output_validator"):
            strict.validate("first\nsecond")
        assert any("strict_mode" in r.message and "multi-line" in r.message for r in caplog.records)

    def test_lenient_multi_line_takes_first_line(self, lenient: OutputValidator) -> None:
        result = lenient.validate("First line of dialogue.\nSome explanation.")
        assert result == "First line of dialogue."

    def test_lenient_skips_empty_first_lines(self, lenient: OutputValidator) -> None:
        """Lenient mode skips blank leading lines to find the first real content."""
        result = lenient.validate("\n\n  \nActual dialogue here.")
        assert result == "Actual dialogue here."

    def test_lenient_all_empty_lines_returns_none(self, lenient: OutputValidator) -> None:
        result = lenient.validate("\n\n   \n\n")
        assert result is None

    def test_single_line_passes_strict(self, strict: OutputValidator) -> None:
        """Single line without newlines is unaffected by step 3."""
        assert strict.validate("Hello, traveller.") == "Hello, traveller."


# ── Step 4: Quote stripping ───────────────────────────────────────────────────


class TestQuoteStripping:
    """Step 4 removes surrounding quotation marks added by some models."""

    def test_double_quotes_stripped(self, strict: OutputValidator) -> None:
        assert strict.validate('"Hello, traveller."') == "Hello, traveller."

    def test_single_quotes_stripped(self, strict: OutputValidator) -> None:
        assert strict.validate("'Hello, traveller.'") == "Hello, traveller."

    def test_no_quotes_unchanged(self, strict: OutputValidator) -> None:
        assert strict.validate("Hello, traveller.") == "Hello, traveller."

    def test_inner_quotes_preserved(self, strict: OutputValidator) -> None:
        """Only the outermost wrapping quotes are stripped."""
        result = strict.validate('"She said "hello" to him."')
        # Outer double quotes stripped, inner ones preserved.
        assert result == 'She said "hello" to him.'

    def test_mismatched_quotes_partially_stripped(self, strict: OutputValidator) -> None:
        """Stripping is performed once per side; mismatched quotes leave one side."""
        result = strict.validate('"Hello.')
        # strip('"') removes leading " but there is no trailing "
        assert result == "Hello."

    def test_empty_double_quotes_returns_none(self, strict: OutputValidator) -> None:
        """'""' after stripping → empty → step 7 returns None."""
        assert strict.validate('""') is None

    def test_whitespace_after_quote_strip_cleaned(self, strict: OutputValidator) -> None:
        """Extra whitespace inside quotes is removed by the final strip()."""
        assert strict.validate('"  Hello.  "') == "Hello."


# ── Step 5: Forbidden pattern check (strict only) ────────────────────────────


class TestForbiddenPatterns:
    """Step 5 rejects structural violations in strict mode only."""

    def test_strict_emote_line_rejected(self, strict: OutputValidator) -> None:
        """Lines fully wrapped in asterisks indicate emotes."""
        assert strict.validate("*sighs deeply*") is None

    def test_strict_stage_direction_rejected(self, strict: OutputValidator) -> None:
        """Bracketed text anywhere in the line is forbidden."""
        assert strict.validate("Hello there. [waves]") is None

    def test_strict_parenthetical_narration_rejected(self, strict: OutputValidator) -> None:
        """Lines entirely wrapped in parentheses indicate narration."""
        assert strict.validate("(The figure steps forward.)") is None

    def test_strict_forbidden_logs_warning(
        self, strict: OutputValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.output_validator"):
            strict.validate("*emote*")
        assert any("strict_mode" in r.message for r in caplog.records)

    def test_lenient_emote_line_passes(self, lenient: OutputValidator) -> None:
        """In lenient mode, structural violations are not checked."""
        result = lenient.validate("*sighs deeply*")
        assert result == "*sighs deeply*"

    def test_lenient_stage_direction_passes(self, lenient: OutputValidator) -> None:
        assert lenient.validate("Hello there. [waves]") == "Hello there. [waves]"

    def test_lenient_parenthetical_passes(self, lenient: OutputValidator) -> None:
        assert lenient.validate("(The figure steps forward.)") is not None

    def test_normal_text_passes_strict(self, strict: OutputValidator) -> None:
        """Plain dialogue without any forbidden patterns passes."""
        assert strict.validate("What brings you to this place?") == "What brings you to this place?"

    def test_asterisk_not_at_both_ends_passes(self, strict: OutputValidator) -> None:
        """Pattern ``^\\*.*\\*$`` requires asterisk at BOTH start and end."""
        result = strict.validate("It's a fine *evening* indeed.")
        assert result == "It's a fine *evening* indeed."


# ── Step 6: Max-length enforcement ───────────────────────────────────────────


class TestMaxLengthEnforcement:
    """Step 6 handles output that exceeds max_output_chars."""

    def test_strict_over_limit_returns_none(self) -> None:
        validator = OutputValidator(strict_mode=True, max_output_chars=20)
        assert validator.validate("A" * 21) is None

    def test_strict_over_limit_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        validator = OutputValidator(strict_mode=True, max_output_chars=10)
        with caplog.at_level(logging.WARNING, logger="app.output_validator"):
            validator.validate("A" * 15)
        assert any("over-length" in r.message for r in caplog.records)

    def test_lenient_over_limit_truncated(self) -> None:
        validator = OutputValidator(strict_mode=False, max_output_chars=20)
        result = validator.validate("A" * 30)
        assert result is not None
        assert len(result) == 20

    def test_lenient_truncation_rstrips_trailing_whitespace(self) -> None:
        """After truncation, trailing whitespace is removed."""
        validator = OutputValidator(strict_mode=False, max_output_chars=10)
        # Position 10 falls in the middle of "   " (spaces after word)
        result = validator.validate("Hello     World")
        assert result == "Hello"

    def test_exactly_at_limit_passes(self) -> None:
        validator = OutputValidator(strict_mode=True, max_output_chars=10)
        assert validator.validate("A" * 10) == "A" * 10

    def test_under_limit_passes(self) -> None:
        validator = OutputValidator(strict_mode=True, max_output_chars=100)
        text = "Short text."
        assert validator.validate(text) == text


# ── Step 7: Final empty check ─────────────────────────────────────────────────


class TestFinalEmptyCheck:
    """Step 7 catches edge cases where earlier steps produced empty strings."""

    def test_empty_after_quote_strip_returns_none(self, strict: OutputValidator) -> None:
        """'""' → after stripping → '' → None."""
        assert strict.validate('""') is None

    def test_single_char_passes(self, strict: OutputValidator) -> None:
        """A single character survives all steps."""
        assert strict.validate("X") == "X"


# ── Integration / happy-path tests ───────────────────────────────────────────


class TestHappyPath:
    """End-to-end: well-formed dialogue should pass unchanged."""

    def test_normal_dialogue_strict(self, strict: OutputValidator) -> None:
        text = "A weathered figure stands near the threshold."
        assert strict.validate(text) == text

    def test_normal_dialogue_lenient(self, lenient: OutputValidator) -> None:
        text = "You look tired, friend. Rest here a while."
        assert lenient.validate(text) == text

    def test_strips_surrounding_whitespace(self, strict: OutputValidator) -> None:
        assert strict.validate("  Hello there.  ") == "Hello there."

    def test_unicode_content_passes(self, strict: OutputValidator) -> None:
        text = "Aye, the path ahead is treacherous — beware."
        assert strict.validate(text) == text

    def test_custom_max_output_chars_respected(self) -> None:
        """A validator constructed with max_output_chars=50 rejects longer text."""
        v = OutputValidator(strict_mode=True, max_output_chars=50)
        assert v.validate("A" * 51) is None
        assert v.validate("A" * 50) is not None

    def test_lenient_recovers_multi_line_with_quotes(self, lenient: OutputValidator) -> None:
        """Lenient mode: take first line and strip its quotes."""
        result = lenient.validate('"Hello, stranger."\nSome additional prose.')
        assert result == "Hello, stranger."

    def test_passthrough_not_affected_by_mode(self) -> None:
        """Both modes reject PASSTHROUGH — it is not a lenient-recoverable violation."""
        for mode in [True, False]:
            v = OutputValidator(strict_mode=mode)
            assert v.validate("PASSTHROUGH") is None


# ── Logging tests ─────────────────────────────────────────────────────────────


class TestLogging:
    """Verify that each rejection path logs at the expected level."""

    def test_passthrough_logs_debug(
        self, strict: OutputValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.output_validator"):
            strict.validate("PASSTHROUGH")
        assert any(r.levelno == logging.DEBUG for r in caplog.records)

    def test_strict_multi_line_logs_warning(
        self, strict: OutputValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.output_validator"):
            strict.validate("line 1\nline 2")
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_strict_forbidden_pattern_logs_warning(
        self, strict: OutputValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="app.output_validator"):
            strict.validate("*emote*")
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_strict_over_length_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        v = OutputValidator(strict_mode=True, max_output_chars=5)
        with caplog.at_level(logging.WARNING, logger="app.output_validator"):
            v.validate("A" * 10)
        assert any("over-length" in r.message for r in caplog.records)

    def test_success_logs_nothing(
        self, strict: OutputValidator, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A clean, valid output should not produce any log records."""
        with caplog.at_level(logging.DEBUG, logger="app.output_validator"):
            result = strict.validate("Hello, traveller.")
        assert result is not None
        assert caplog.records == []
