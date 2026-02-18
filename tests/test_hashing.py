"""
Tests for app/hashing.py — normalisation and hash utility functions.

Each test class targets a single function.  The tests are organised to verify:
  1. Correctness of normalisation rules (edge cases, boundary conditions).
  2. Determinism (same input always produces same output).
  3. Sensitivity (different inputs produce different outputs).
  4. Format (64-character lowercase hex digest where applicable).
"""

from __future__ import annotations

from app.hashing import (
    _normalise_output,
    _normalise_system_prompt,
    compute_ipc_id,
    compute_output_hash,
    compute_payload_hash,
    compute_system_prompt_hash,
)

# ── _normalise_system_prompt ────────────────────────────────────────────────


class TestNormaliseSystemPrompt:
    """Verify normalisation rules for system prompt text."""

    def test_strips_leading_trailing_whitespace_per_line(self) -> None:
        """Each line should have its leading and trailing spaces removed."""
        raw = "  line one  \n  line two  "
        assert _normalise_system_prompt(raw) == "line one\nline two"

    def test_removes_blank_lines_at_start_and_end(self) -> None:
        """Blank lines at the very start and end of the text are removed."""
        raw = "\n\n  content here  \n\n"
        assert _normalise_system_prompt(raw) == "content here"

    def test_preserves_internal_blank_lines(self) -> None:
        """Blank lines between content lines are preserved (paragraph breaks)."""
        raw = "paragraph one\n\nparagraph two"
        result = _normalise_system_prompt(raw)
        assert result == "paragraph one\n\nparagraph two"

    def test_preserves_case(self) -> None:
        """Case is semantic — normalisation must never lowercase."""
        raw = "NEVER use metaphor"
        assert _normalise_system_prompt(raw) == "NEVER use metaphor"

    def test_preserves_line_order(self) -> None:
        """Lines must remain in their original order after normalisation."""
        raw = "first\nsecond\nthird"
        assert _normalise_system_prompt(raw) == "first\nsecond\nthird"

    def test_empty_string(self) -> None:
        """An empty input should produce an empty normalised string."""
        assert _normalise_system_prompt("") == ""

    def test_whitespace_only(self) -> None:
        """A whitespace-only input should normalise to empty string."""
        assert _normalise_system_prompt("   \n  \n   ") == ""

    def test_tabs_stripped(self) -> None:
        """Tab characters in leading/trailing whitespace are stripped."""
        raw = "\tindented line\t"
        assert _normalise_system_prompt(raw) == "indented line"


# ── _normalise_output ───────────────────────────────────────────────────────


class TestNormaliseOutput:
    """Verify normalisation rules for LLM output text."""

    def test_strips_outer_whitespace(self) -> None:
        """Leading and trailing whitespace on the entire string is removed."""
        raw = "  some output text  "
        assert _normalise_output(raw) == "some output text"

    def test_collapses_multiple_spaces(self) -> None:
        """Runs of 2+ spaces are collapsed to a single space."""
        raw = "word   word    word"
        assert _normalise_output(raw) == "word word word"

    def test_preserves_single_spaces(self) -> None:
        """Single spaces between words are not altered."""
        raw = "word word word"
        assert _normalise_output(raw) == "word word word"

    def test_preserves_punctuation_and_case(self) -> None:
        """Punctuation and letter casing must be preserved exactly."""
        raw = "Hello, World! It's a TEST."
        assert _normalise_output(raw) == "Hello, World! It's a TEST."

    def test_preserves_newlines(self) -> None:
        """Newline characters are preserved (only spaces are collapsed)."""
        raw = "line one\nline two"
        assert _normalise_output(raw) == "line one\nline two"

    def test_empty_string(self) -> None:
        """An empty input should produce an empty normalised string."""
        assert _normalise_output("") == ""

    def test_whitespace_only(self) -> None:
        """A whitespace-only input should normalise to empty string."""
        assert _normalise_output("   ") == ""


# ── compute_system_prompt_hash ──────────────────────────────────────────────


class TestComputeSystemPromptHash:
    """Verify SHA-256 hashing of normalised system prompts."""

    def test_deterministic(self) -> None:
        """Same input text must always produce the same hash."""
        prompt = "You are a descriptive layer."
        assert compute_system_prompt_hash(prompt) == compute_system_prompt_hash(prompt)

    def test_returns_64_char_hex(self) -> None:
        """The digest must be a 64-character lowercase hex string."""
        result = compute_system_prompt_hash("test prompt")
        assert len(result) == 64
        assert result == result.lower()
        # Verify it's valid hex by parsing it
        int(result, 16)

    def test_different_prompts_differ(self) -> None:
        """Semantically different prompts must produce different hashes."""
        h1 = compute_system_prompt_hash("prompt version A")
        h2 = compute_system_prompt_hash("prompt version B")
        assert h1 != h2

    def test_whitespace_variations_same_hash(self) -> None:
        """Prompts differing only in insignificant whitespace should hash identically."""
        # Leading/trailing spaces on lines, extra blank lines at edges
        raw_a = "  line one  \n  line two  "
        raw_b = "line one\nline two"
        assert compute_system_prompt_hash(raw_a) == compute_system_prompt_hash(raw_b)

    def test_case_sensitivity(self) -> None:
        """Prompts differing in case must produce different hashes."""
        h_upper = compute_system_prompt_hash("NEVER use metaphor")
        h_lower = compute_system_prompt_hash("never use metaphor")
        assert h_upper != h_lower


# ── compute_output_hash ─────────────────────────────────────────────────────


class TestComputeOutputHash:
    """Verify SHA-256 hashing of normalised LLM output text."""

    def test_deterministic(self) -> None:
        """Same output text must always produce the same hash."""
        text = "A weathered figure stands."
        assert compute_output_hash(text) == compute_output_hash(text)

    def test_returns_64_char_hex(self) -> None:
        """The digest must be a 64-character lowercase hex string."""
        result = compute_output_hash("test output")
        assert len(result) == 64
        assert result == result.lower()
        int(result, 16)

    def test_different_outputs_differ(self) -> None:
        """Different output texts must produce different hashes."""
        h1 = compute_output_hash("output A")
        h2 = compute_output_hash("output B")
        assert h1 != h2

    def test_extra_spaces_normalised(self) -> None:
        """Outputs differing only in extra spaces should hash identically."""
        raw_a = "word  word"
        raw_b = "word word"
        assert compute_output_hash(raw_a) == compute_output_hash(raw_b)

    def test_case_sensitivity(self) -> None:
        """Outputs differing in case must produce different hashes."""
        h_upper = compute_output_hash("HELLO world")
        h_lower = compute_output_hash("hello world")
        assert h_upper != h_lower


# ── compute_ipc_id ──────────────────────────────────────────────────────────


class TestComputeIpcId:
    """Verify the Interpretive Provenance Chain (IPC) identifier."""

    # Baseline kwargs used across tests — a consistent set of provenance fields.
    _BASE = {
        "input_hash": "a" * 64,
        "system_prompt_hash": "b" * 64,
        "model": "gemma2:2b",
        "temperature": 0.2,
        "max_tokens": 120,
        "seed": 42,
    }

    def test_deterministic(self) -> None:
        """Same provenance fields must always produce the same IPC ID."""
        assert compute_ipc_id(**self._BASE) == compute_ipc_id(**self._BASE)

    def test_returns_64_char_hex(self) -> None:
        """The IPC ID must be a 64-character lowercase hex string."""
        result = compute_ipc_id(**self._BASE)
        assert len(result) == 64
        assert result == result.lower()
        int(result, 16)

    def test_different_inputs_differ(self) -> None:
        """Changing any field must produce a different IPC ID."""
        base_id = compute_ipc_id(**self._BASE)
        modified = {**self._BASE, "input_hash": "c" * 64}
        assert compute_ipc_id(**modified) != base_id

    def test_all_fields_affect_hash(self) -> None:
        """Each individual field must influence the final IPC ID."""
        base_id = compute_ipc_id(**self._BASE)

        # Change each field one at a time and verify the hash changes
        variations = [
            {"input_hash": "x" * 64},
            {"system_prompt_hash": "x" * 64},
            {"model": "llama3:8b"},
            {"temperature": 0.9},
            {"max_tokens": 256},
            {"seed": 999},
        ]
        for override in variations:
            modified = {**self._BASE, **override}
            assert (
                compute_ipc_id(**modified) != base_id
            ), f"Changing {list(override.keys())[0]} did not change the IPC ID"

    def test_colon_delimiter_prevents_collision(self) -> None:
        """
        The colon delimiter must prevent collisions from field concatenation.

        Without a delimiter, input_hash="ab" + system_prompt_hash="cd" would
        produce the same concatenation as input_hash="abc" + system_prompt_hash="d".
        The colon separator ("ab:cd" vs "abc:d") prevents this.
        """
        id_a = compute_ipc_id(
            input_hash="ab",
            system_prompt_hash="cd",
            model="m",
            temperature=0.1,
            max_tokens=10,
            seed=1,
        )
        id_b = compute_ipc_id(
            input_hash="abc",
            system_prompt_hash="d",
            model="m",
            temperature=0.1,
            max_tokens=10,
            seed=1,
        )
        assert id_a != id_b


# ── compute_payload_hash ────────────────────────────────────────────────────


class TestComputePayloadHash:
    """Verify SHA-256 hashing of canonical payload dictionaries."""

    _SAMPLE = {
        "axes": {
            "health": {"label": "weary", "score": 0.5},
            "age": {"label": "old", "score": 0.7},
        },
        "policy_hash": "abc123",
        "seed": 42,
        "world_id": "test_world",
    }

    def test_deterministic(self) -> None:
        """Same dict must always produce the same hash."""
        assert compute_payload_hash(self._SAMPLE) == compute_payload_hash(self._SAMPLE)

    def test_order_independent(self) -> None:
        """Dict key insertion order must not affect the hash."""
        # Build the same dict with reversed key order
        reversed_axes = {
            "age": {"label": "old", "score": 0.7},
            "health": {"label": "weary", "score": 0.5},
        }
        reversed_dict = {
            "world_id": "test_world",
            "seed": 42,
            "policy_hash": "abc123",
            "axes": reversed_axes,
        }
        assert compute_payload_hash(self._SAMPLE) == compute_payload_hash(reversed_dict)

    def test_returns_64_char_hex(self) -> None:
        """The digest must be a 64-character lowercase hex string."""
        result = compute_payload_hash(self._SAMPLE)
        assert len(result) == 64
        assert result == result.lower()
        int(result, 16)

    def test_different_payloads_differ(self) -> None:
        """Different payload content must produce different hashes."""
        modified = {**self._SAMPLE, "seed": 999}
        assert compute_payload_hash(self._SAMPLE) != compute_payload_hash(modified)
