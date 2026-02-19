"""
app/micro_indicators.py
-----------------------------------------------------------------------------
Micro-Indicators — Structural Pattern Classification for Transformation Map
Rows.

Why a dedicated module?
-----------------------
The Transformation Map (``transformation_map.py``) extracts clause-level
replacement pairs (removed/added) between two texts.  These pairs reveal
*what* changed, but not the *structural character* of the change.

Micro-indicators fill that gap by labelling each row with one or more
deterministic heuristic tags — "compression", "embodiment shift",
"intensity ↑", etc. — that surface structural writing patterns without
performing semantic interpretation.

The 10 indicators
-----------------
1. **compression**       — removed tokens ≥ ratio × added tokens
2. **expansion**         — added tokens ≥ ratio × removed tokens
3. **embodiment shift**  — abstract words removed, physical words added
4. **abstraction ↑**     — concrete words removed, abstract words added
5. **intensity ↑**       — word moves up on a known intensity scale
6. **intensity ↓**       — word moves down on a known intensity scale
7. **consolidation**     — sentence count decreases
8. **fragmentation**     — sentence count increases
9. **tone reframing**    — lexical substitution with no other structural
   shift (fallback)
10. **modality shift**   — verb/adjective density change (POS tagging)
11. **lexical pivot**    — rare content word → rare content word (fallback)

Design principles
-----------------
• **Deterministic**: same input always produces the same indicators.
• **Rule-based**: no AI inference, no embeddings, no probabilistic reasoning.
• **Conservative**: defaults are tuned to avoid false positives.
• **Educational**: labels introduce structural writing vocabulary.
• **Transparent**: each heuristic is a simple, inspectable rule.

Lexicon data
------------
Three JSON files in ``app/data/`` provide the vocabulary for lexicon-based
indicators:

- ``embodiment_v0_1.json`` — abstract/physical word lists
- ``abstraction_v0_1.json`` — concrete/abstract term lists
- ``intensity_v0_1.json`` — ordered intensity scales

These are loaded once at module import time and converted to ``frozenset``
lookups for O(1) membership testing.

NLTK data requirements
----------------------
Reuses the same NLTK data packages as ``signal_isolation.py`` (punkt_tab,
stopwords, wordnet) which are ensured at import time via the side-effect
import of ``app.signal_isolation``.

Additionally requires ``averaged_perceptron_tagger_eng`` for the modality
shift indicator (POS tagging).  This is downloaded automatically via
``_ensure_pos_tagger_data()`` at module load time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Side-effect import: ensures punkt_tab, stopwords, and wordnet are
# available before this module tries to use them.
import app.signal_isolation  # noqa: F401

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# NLTK data bootstrap (POS tagger)
# -----------------------------------------------------------------------------

# The modality shift indicator requires POS tagging.  This data package
# is not needed by signal_isolation.py, so we download it separately.
_REQUIRED_EXTRA_NLTK: tuple[tuple[str, str], ...] = (
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
)


def _ensure_pos_tagger_data() -> None:
    """Ensure the POS tagger data is available for modality shift detection."""
    for pkg_name, find_path in _REQUIRED_EXTRA_NLTK:
        try:
            nltk.data.find(find_path)
        except LookupError:
            logger.info("Downloading NLTK data package: %s", pkg_name)
            try:
                nltk.download(pkg_name, quiet=True)
            except Exception as exc:  # noqa: BLE001 – intentionally broad
                logger.warning(
                    "Failed to download NLTK '%s': %s: %s",
                    pkg_name,
                    type(exc).__name__,
                    exc,
                )


# Run once at module import time.
_ensure_pos_tagger_data()


# -----------------------------------------------------------------------------
# Lexicon data loading
# -----------------------------------------------------------------------------

_DATA_DIR: Path = Path(__file__).parent / "data"


def _load_json(filename: str) -> dict:
    """Load a JSON file from the ``app/data/`` directory."""
    path = _DATA_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Load lexicon data once at module import time.
_EMBODIMENT_DATA: dict = _load_json("embodiment_v0_1.json")
_ABSTRACTION_DATA: dict = _load_json("abstraction_v0_1.json")
_INTENSITY_DATA: dict = _load_json("intensity_v0_1.json")

# Pre-compute frozensets for O(1) membership testing.
# Embodiment lexicon: abstract ↔ physical (for embodiment shift)
_ABSTRACT_WORDS: frozenset[str] = frozenset(w.lower() for w in _EMBODIMENT_DATA["abstract"])
_PHYSICAL_WORDS: frozenset[str] = frozenset(w.lower() for w in _EMBODIMENT_DATA["physical"])

# Abstraction lexicon: concrete ↔ abstract (for abstraction ↑)
_ABSTRACT_TERMS: frozenset[str] = frozenset(w.lower() for w in _ABSTRACTION_DATA["abstract_terms"])
_CONCRETE_TERMS: frozenset[str] = frozenset(w.lower() for w in _ABSTRACTION_DATA["concrete_terms"])

# Intensity index: word → list of (scale_name, position_index).
# A word may appear on multiple scales (though this is unlikely in v0.1).
_INTENSITY_INDEX: dict[str, list[tuple[str, int]]] = {}
for _scale_name, _scale_words in _INTENSITY_DATA["scales"].items():
    for _idx, _word in enumerate(_scale_words):
        _key = _word.lower()
        _INTENSITY_INDEX.setdefault(_key, []).append((_scale_name, _idx))

# English stopwords for lexical pivot detection.
_ENGLISH_STOPWORDS: frozenset[str] = frozenset(stopwords.words("english"))

# Union of all known lexicon words — used to identify "rare" words for
# the lexical pivot indicator.
_ALL_KNOWN_LEXICON: frozenset[str] = (
    _ABSTRACT_WORDS | _PHYSICAL_WORDS | _ABSTRACT_TERMS | _CONCRETE_TERMS
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IndicatorConfig:
    """
    Tuning parameters for micro-indicator detection.

    All fields have conservative defaults.  The frontend can override
    these per-request via the ``indicator_config`` field on the
    ``TransformationMapRequest`` schema.

    Parameters
    ----------
    compression_ratio : float
        Minimum ratio of ``len(removed_tokens) / len(added_tokens)`` to
        flag "compression".  Default 2.0 means removed must be at least
        twice as long as added.
    expansion_ratio : float
        Minimum ratio of ``len(added_tokens) / len(removed_tokens)`` to
        flag "expansion".  Default 2.0.
    min_tokens : int
        Minimum token count on the *larger* side to consider size-based
        indicators (compression/expansion).  Prevents flagging single-word
        swaps.  Default 2.
    modality_density_threshold : float
        Minimum absolute change in verb+adjective density (proportion of
        tokens that are verbs or adjectives) to flag "modality shift".
        Default 0.3 (conservative — requires a 30 percentage-point shift).
    enabled : tuple[str, ...] | None
        When not None, only compute indicators whose names appear in this
        tuple.  None means all indicators are active.
    """

    compression_ratio: float = 2.0
    expansion_ratio: float = 2.0
    min_tokens: int = 2
    modality_density_threshold: float = 0.3
    enabled: tuple[str, ...] | None = None


# Canonical ordered list of all indicator names.
ALL_INDICATORS: list[str] = [
    "compression",
    "expansion",
    "embodiment shift",
    "abstraction \u2191",
    "intensity \u2191",
    "intensity \u2193",
    "consolidation",
    "fragmentation",
    "tone reframing",
    "modality shift",
    "lexical pivot",
]


# Default configuration instance (avoids re-creating on every call).
_DEFAULT_CONFIG: IndicatorConfig = IndicatorConfig()


# -----------------------------------------------------------------------------
# Private tokenisation helper
# -----------------------------------------------------------------------------


def _tokenize_lower(text: str) -> list[str]:
    """
    Tokenize text and return lowercase tokens containing at least one
    alphabetic character.

    Uses NLTK's Penn Treebank tokeniser (same as ``signal_isolation.py``).
    Discards punctuation-only and numeric-only tokens.

    Parameters
    ----------
    text : str
        Raw text to tokenize.

    Returns
    -------
    list[str]
        Lowercase alpha-containing tokens, in order.
    """
    return [t.lower() for t in word_tokenize(text) if any(c.isalpha() for c in t)]


# -----------------------------------------------------------------------------
# Individual indicator classifiers
# -----------------------------------------------------------------------------


def _check_compression(
    removed_tokens: list[str],
    added_tokens: list[str],
    config: IndicatorConfig,
) -> str | None:
    """
    Check for compression: many tokens condensed into fewer tokens.

    Returns ``"compression"`` if ``len(removed) >= ratio * len(added)``
    and the larger side has at least ``min_tokens`` tokens.
    """
    if not added_tokens or len(removed_tokens) < config.min_tokens:
        return None
    if len(removed_tokens) >= config.compression_ratio * len(added_tokens):
        return "compression"
    return None


def _check_expansion(
    removed_tokens: list[str],
    added_tokens: list[str],
    config: IndicatorConfig,
) -> str | None:
    """
    Check for expansion: short phrase rewritten into longer clause.

    Returns ``"expansion"`` if ``len(added) >= ratio * len(removed)``
    and the larger side has at least ``min_tokens`` tokens.
    """
    if not removed_tokens or len(added_tokens) < config.min_tokens:
        return None
    if len(added_tokens) >= config.expansion_ratio * len(removed_tokens):
        return "expansion"
    return None


def _check_embodiment_shift(
    removed_tokens: list[str],
    added_tokens: list[str],
) -> str | None:
    """
    Check for embodiment shift: abstract → physical.

    Returns ``"embodiment shift"`` if at least one removed token is in
    the abstract lexicon AND at least one added token is in the physical
    lexicon.

    Uses the ``embodiment_v0_1.json`` word lists.
    """
    removed_set = set(removed_tokens)
    added_set = set(added_tokens)
    has_abstract_removed = bool(removed_set & _ABSTRACT_WORDS)
    has_physical_added = bool(added_set & _PHYSICAL_WORDS)
    if has_abstract_removed and has_physical_added:
        return "embodiment shift"
    return None


def _check_abstraction_up(
    removed_tokens: list[str],
    added_tokens: list[str],
) -> str | None:
    """
    Check for abstraction increase: concrete → abstract.

    Returns ``"abstraction ↑"`` if at least one removed token is in
    the concrete lexicon AND at least one added token is in the abstract
    lexicon.

    Uses the ``abstraction_v0_1.json`` word lists.
    """
    removed_set = set(removed_tokens)
    added_set = set(added_tokens)
    has_concrete_removed = bool(removed_set & _CONCRETE_TERMS)
    has_abstract_added = bool(added_set & _ABSTRACT_TERMS)
    if has_concrete_removed and has_abstract_added:
        return "abstraction \u2191"
    return None


def _check_intensity(
    removed_tokens: list[str],
    added_tokens: list[str],
) -> str | None:
    """
    Check for intensity shift: word moves up or down a known scale.

    Returns ``"intensity ↑"`` or ``"intensity ↓"`` if a removed token
    and an added token appear on the *same* intensity scale at different
    positions.

    Uses the ``intensity_v0_1.json`` scale data.  If multiple scale
    matches are found, the first match wins.
    """
    removed_set = set(removed_tokens)
    added_set = set(added_tokens)

    for word_r in removed_set:
        if word_r not in _INTENSITY_INDEX:
            continue
        for scale_name, idx_r in _INTENSITY_INDEX[word_r]:
            for word_a in added_set:
                if word_a not in _INTENSITY_INDEX:
                    continue
                for sn_a, idx_a in _INTENSITY_INDEX[word_a]:
                    if sn_a == scale_name and idx_a != idx_r:
                        return "intensity \u2191" if idx_a > idx_r else "intensity \u2193"
    return None


def _check_consolidation(removed: str, added: str) -> str | None:
    """
    Check for consolidation: multiple sentences merged into fewer.

    Returns ``"consolidation"`` if the removed text contains more
    sentences than the added text (and the removed text has at least 2
    sentences).

    Operates on raw strings (not pre-tokenized) because sentence
    splitting requires the original punctuation context.
    """
    sents_r = sent_tokenize(removed)
    sents_a = sent_tokenize(added)
    if len(sents_r) > 1 and len(sents_a) < len(sents_r):
        return "consolidation"
    return None


def _check_fragmentation(removed: str, added: str) -> str | None:
    """
    Check for fragmentation: single clause split into multiple sentences.

    Returns ``"fragmentation"`` if the added text contains more sentences
    than the removed text (and the added text has at least 2 sentences).
    """
    sents_r = sent_tokenize(removed)
    sents_a = sent_tokenize(added)
    if len(sents_a) > 1 and len(sents_a) > len(sents_r):
        return "fragmentation"
    return None


def _check_modality_shift(
    removed_tokens: list[str],
    added_tokens: list[str],
    config: IndicatorConfig,
) -> str | None:
    """
    Check for modality shift: significant change in verb/adjective density.

    Returns ``"modality shift"`` if the absolute difference in
    verb+adjective density between removed and added sides exceeds the
    configured threshold.

    Uses NLTK POS tagging.  This is the most computationally expensive
    indicator and is intentionally conservative (high default threshold).
    """
    if not removed_tokens or not added_tokens:
        return None

    try:
        pos_r = nltk.pos_tag(removed_tokens)
        pos_a = nltk.pos_tag(added_tokens)
    except Exception:  # noqa: BLE001 – graceful degradation
        return None

    # Penn Treebank POS tags for verbs and adjectives.
    _VA_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS"}

    va_r = sum(1 for _, tag in pos_r if tag in _VA_TAGS)
    va_a = sum(1 for _, tag in pos_a if tag in _VA_TAGS)

    density_r = va_r / len(removed_tokens)
    density_a = va_a / len(added_tokens)

    if abs(density_a - density_r) >= config.modality_density_threshold:
        return "modality shift"
    return None


def _check_lexical_pivot(
    removed_tokens: list[str],
    added_tokens: list[str],
) -> str | None:
    """
    Check for lexical pivot: rare content word replaced by another rare
    content word.

    A "rare" word is one that:
    - Is not an English stopword
    - Does not appear in any of the known lexicon sets

    This is a fallback indicator — it fires only when no other structural
    indicator matched, catching meaningful word substitutions that don't
    fit the other categories.
    """
    rare_removed = [
        t for t in removed_tokens if t not in _ENGLISH_STOPWORDS and t not in _ALL_KNOWN_LEXICON
    ]
    rare_added = [
        t for t in added_tokens if t not in _ENGLISH_STOPWORDS and t not in _ALL_KNOWN_LEXICON
    ]
    if rare_removed and rare_added:
        return "lexical pivot"
    return None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def classify_row(
    removed: str,
    added: str,
    *,
    config: IndicatorConfig | None = None,
) -> list[str]:
    """
    Compute micro-indicators for a single transformation map row.

    Evaluates all applicable indicator heuristics against the removed/added
    text pair and returns a list of indicator labels.  A row can have zero
    or more indicators (e.g., ``["compression", "intensity ↑"]``).

    Structural indicators are evaluated first; fallback indicators
    (``tone reframing``, ``lexical pivot``) only fire when no structural
    indicator matched.

    Parameters
    ----------
    removed : str
        The text chunk from the baseline (A) that was replaced.
    added : str
        The text chunk from the current text (B) that replaced it.
    config : IndicatorConfig | None
        Optional tuning parameters.  ``None`` uses conservative defaults.

    Returns
    -------
    list[str]
        Ordered list of indicator labels that apply to this row.
        Empty list if no indicators match or if both inputs are empty.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    enabled = set(config.enabled) if config.enabled is not None else None

    def _is_enabled(name: str) -> bool:
        return enabled is None or name in enabled

    # Skip classification for empty or purely whitespace inputs.
    if not removed.strip() and not added.strip():
        return []

    removed_tokens = _tokenize_lower(removed) if removed.strip() else []
    added_tokens = _tokenize_lower(added) if added.strip() else []

    indicators: list[str] = []

    # -- Structural size indicators (mutually exclusive pair) ---------------

    if _is_enabled("compression"):
        result = _check_compression(removed_tokens, added_tokens, config)
        if result:
            indicators.append(result)

    # Only check expansion if compression didn't fire (mutually exclusive).
    if not indicators and _is_enabled("expansion"):
        result = _check_expansion(removed_tokens, added_tokens, config)
        if result:
            indicators.append(result)

    # -- Sentence boundary indicators (mutually exclusive pair) -------------

    if _is_enabled("consolidation"):
        result = _check_consolidation(removed, added)
        if result:
            indicators.append(result)

    if "consolidation" not in indicators and _is_enabled("fragmentation"):
        result = _check_fragmentation(removed, added)
        if result:
            indicators.append(result)

    # -- Lexicon-based indicators (embodiment/abstraction mutually excl.) ---

    if _is_enabled("embodiment shift"):
        result = _check_embodiment_shift(removed_tokens, added_tokens)
        if result:
            indicators.append(result)

    # Only check abstraction if embodiment didn't fire.
    if "embodiment shift" not in indicators and _is_enabled("abstraction \u2191"):
        result = _check_abstraction_up(removed_tokens, added_tokens)
        if result:
            indicators.append(result)

    # -- Intensity (independent — can co-occur with other indicators) -------

    if _is_enabled("intensity \u2191") or _is_enabled("intensity \u2193"):
        result = _check_intensity(removed_tokens, added_tokens)
        if result and _is_enabled(result):
            indicators.append(result)

    # -- POS-based indicator (conservative) ---------------------------------

    if _is_enabled("modality shift"):
        result = _check_modality_shift(removed_tokens, added_tokens, config)
        if result:
            indicators.append(result)

    # -- Fallback: tone reframing ------------------------------------------
    # Fires when there is a lexical substitution but no structural
    # indicator matched.  "Something changed, but we can't classify the
    # structural character of the change."

    if not indicators and _is_enabled("tone reframing"):
        if removed_tokens and added_tokens and set(removed_tokens) != set(added_tokens):
            indicators.append("tone reframing")

    # -- Fallback: lexical pivot -------------------------------------------
    # Fires only as a secondary fallback — when no other indicator matched
    # and tone reframing also didn't fire (i.e., the token sets are the
    # same but rare content words differ).

    if not indicators and _is_enabled("lexical pivot"):
        result = _check_lexical_pivot(removed_tokens, added_tokens)
        if result:
            indicators.append(result)

    return indicators


def classify_rows(
    rows: list[dict[str, str]],
    *,
    config: IndicatorConfig | None = None,
) -> list[list[str]]:
    """
    Compute micro-indicators for every row in a transformation map.

    Convenience wrapper that calls :func:`classify_row` on each row.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Each dict must have ``"removed"`` and ``"added"`` keys (the
        output of :func:`~app.transformation_map.compute_transformation_map`).
    config : IndicatorConfig | None
        Optional tuning parameters.  ``None`` uses conservative defaults.

    Returns
    -------
    list[list[str]]
        One list of indicator labels per row, in the same order as the
        input rows.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    return [classify_row(row["removed"], row["added"], config=config) for row in rows]
