"""
app/output_validator.py
-----------------------------------------------------------------------------
Output validator for the OOC→IC translation layer.

Ported from mud_server's translation/validator.py so the lab is fully
independent of the mud_server package.

Validation pipeline (applied in order)
---------------------------------------
Each step either passes the text through to the next step, mutates it
(steps 3 and 6 in lenient mode), or rejects it (returns None).

1. **Empty check** — Reject blank/whitespace-only strings immediately.
2. **PASSTHROUGH sentinel** — The LLM is instructed to return the literal
   string "PASSTHROUGH" when the OOC message cannot be sensibly translated
   into IC dialogue (e.g. purely mechanical commands, out-of-game requests).
   Any text that starts with "PASSTHROUGH" (case-insensitive) is treated as
   a deliberate no-translation signal and rejected here.
3. **Multi-line check** — IC dialogue must be a single line.
   - *strict_mode=True*: any ``\\n`` in the text → reject.
   - *strict_mode=False*: extract the first non-empty line and continue.
4. **Quote stripping** — Some models (gemma2, llama3) wrap their output in
   double or single quotation marks.  Both are stripped unconditionally.
5. **Forbidden pattern check** — Strict mode only.  Rejects text that
   matches structural patterns indicating the model produced emote lines,
   stage directions, or parenthetical narration instead of raw spoken
   dialogue:
   - ``*emote lines*``
   - ``[stage directions]``
   - ``(parenthetical narration)``
6. **Max-length enforcement** — Output longer than ``max_output_chars``.
   - *strict_mode=True*: reject.
   - *strict_mode=False*: truncate to the limit and right-strip whitespace.
7. **Final empty check** — A second empty guard in case earlier steps
   produced an empty string (e.g. quote-stripping ``""``).

Mode summary
------------
+-------------------+------------------+---------------------+
| Constraint        | strict_mode=True | strict_mode=False   |
+===================+==================+=====================+
| Multi-line        | reject           | take first line     |
+-------------------+------------------+---------------------+
| Forbidden pattern | reject           | pass through        |
+-------------------+------------------+---------------------+
| Over max-length   | reject           | truncate            |
+-------------------+------------------+---------------------+
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel and forbidden-pattern constants
# ---------------------------------------------------------------------------

#: The LLM returns this sentinel (upper-cased) to signal that the OOC message
#: cannot be translated.  Validated case-insensitively via ``.upper()``.
PASSTHROUGH_SENTINEL = "PASSTHROUGH"

#: Compiled patterns whose presence in the output indicates structural
#: constraint violations (i.e. the model produced something other than a
#: bare line of spoken dialogue).  Only enforced in strict mode.
#:
#: Pattern details:
#:   - ``^\*.*\*$``  — lines that start and end with ``*``, e.g. ``*sighs*``
#:   - ``\[.*\]``    — any ``[bracketed]`` text anywhere in the line
#:   - ``^\(.*\)$``  — lines that are entirely wrapped in parentheses
_FORBIDDEN_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\*.*\*$"),   # *emote lines*
    re.compile(r"\[.*\]"),     # [stage directions]
    re.compile(r"^\(.*\)$"),   # (parenthetical narration)
]


class OutputValidator:
    """Validates and cleans raw LLM output before it is stored or displayed.

    The validator is a pure function of its arguments: it holds no mutable
    state and produces the same output for the same input every time.

    The 7-step pipeline is described in the module docstring.  Two operating
    modes are available:

    - **strict_mode=True** (production default): any constraint violation
      returns ``None``, causing the system to fall back to the original OOC
      message.  This prevents any malformed LLM output from reaching players.
    - **strict_mode=False** (lenient / lab default): minor violations (multi-
      line output, over-length text) are corrected rather than rejected.
      Useful for exploring what the LLM would produce with lower guardrails.

    Args:
        strict_mode:      When ``True`` any constraint violation → ``None``.
                          When ``False`` recoverable violations are fixed up.
        max_output_chars: Hard ceiling on IC output character count.
                          Range: typically 50–2000; production default is 280.
    """

    def __init__(self, *, strict_mode: bool = True, max_output_chars: int = 280) -> None:
        self._strict_mode = strict_mode
        self._max_output_chars = max_output_chars

    def validate(self, ic_raw: str) -> str | None:
        """Validate and clean a raw LLM response string.

        Applies the 7-step pipeline described in the module docstring.
        Each step either returns ``None`` (rejection), mutates the working
        copy of the text, or passes it to the next step unchanged.

        Args:
            ic_raw: Raw string returned by the LLM.  May be empty, contain
                    newlines, leading/trailing whitespace, or forbidden
                    patterns.

        Returns:
            A clean, single-line IC string on success; ``None`` on any
            rejection.  The caller should treat ``None`` as a signal to
            fall back to the original OOC message (PASSTHROUGH behaviour).

        Note:
            ``ic_raw`` is first stripped of leading/trailing whitespace
            before any subsequent checks are applied.  The returned string
            is never the original ``ic_raw``; it is always a new value.
        """
        # ── 1. Empty check ────────────────────────────────────────────────────
        # Reject entirely blank or whitespace-only strings before any other
        # processing.  This is separate from step 7 because it runs on the
        # raw, un-stripped value and avoids spurious log messages from later
        # pipeline steps.
        if not ic_raw or not ic_raw.strip():
            return None

        text = ic_raw.strip()

        # ── 2. PASSTHROUGH sentinel ────────────────────────────────────────────
        # The system prompt instructs the model to return "PASSTHROUGH" when
        # the OOC message is untranslatable.  Any response that begins with
        # this sentinel (case-insensitive) is treated as a deliberate signal.
        if text.upper().startswith(PASSTHROUGH_SENTINEL):
            logger.debug("OutputValidator: PASSTHROUGH sentinel returned by model.")
            return None

        # ── 3. Multi-line check ───────────────────────────────────────────────
        # IC dialogue is expected to be exactly one line.  Some models return
        # explanatory prose before or after the dialogue line.
        if "\n" in text:
            if self._strict_mode:
                logger.warning("OutputValidator: strict_mode rejected multi-line output.")
                return None
            # Lenient path: take the first non-empty line.
            first_line = next(
                (line.strip() for line in text.splitlines() if line.strip()), ""
            )
            if not first_line:
                return None
            text = first_line

        # ── 4. Quote stripping ────────────────────────────────────────────────
        # Some models (e.g. gemma2:2b) wrap dialogue in quotation marks:
        # "Hello, traveller."  →  Hello, traveller.
        # Strip both double and single quotes from both ends unconditionally.
        text = text.strip('"').strip("'").strip()

        # ── 5. Forbidden pattern check (strict mode only) ─────────────────────
        # In strict mode, reject output that looks like emotes, stage
        # directions, or parenthetical narration — all of which indicate the
        # model produced something other than raw spoken dialogue.
        if self._strict_mode:
            for pattern in _FORBIDDEN_PATTERNS:
                if pattern.search(text):
                    logger.warning(
                        "OutputValidator: strict_mode rejected output matching pattern %r: %r",
                        pattern.pattern,
                        text[:60],
                    )
                    return None

        # ── 6. Max-length enforcement ─────────────────────────────────────────
        # Long outputs can overflow UI elements and indicate the model
        # ignored the brevity constraint in the system prompt.
        if len(text) > self._max_output_chars:
            if self._strict_mode:
                logger.warning(
                    "OutputValidator: strict_mode rejected over-length output (%d > %d).",
                    len(text),
                    self._max_output_chars,
                )
                return None
            # Lenient path: truncate and clean trailing whitespace.
            text = text[: self._max_output_chars].rstrip()

        # ── 7. Final empty check ──────────────────────────────────────────────
        # Guard against edge cases where earlier steps (e.g. quote-stripping
        # the string '""') produced an empty string.
        return text if text else None
