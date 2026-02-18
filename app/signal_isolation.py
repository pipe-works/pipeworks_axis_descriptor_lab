"""
app/signal_isolation.py
-----------------------------------------------------------------------------
Signal Isolation Layer for the Axis Descriptor Lab.

Why a dedicated module?
-----------------------
The signal isolation pipeline transforms raw LLM output text into a filtered
set of content lemmas so that meaningful lexical pivots can be surfaced
without structural noise.  Centralising this logic in one module keeps
``main.py`` focused on routing and ensures the NLP pipeline is independently
testable.

Pipeline
--------
The module applies a five-step pipeline to each text:

1. **Tokenise** — split text into word tokens using NLTK's Penn Treebank
   tokeniser (``word_tokenize``).  Lowercase all tokens and discard any
   that contain no alphabetic characters (punctuation, numbers).
2. **Lemmatise** — reduce inflected forms to their base lemma using the
   WordNet lemmatiser.  A two-pass heuristic is used: try verb
   lemmatisation first (catches "carries" → "carry", "failing" → "fail"),
   then fall back to the default noun lemmatisation ("figures" → "figure").
3. **Filter stopwords** — remove English function words (articles,
   auxiliaries, pronouns, conjunctions) using NLTK's stopwords corpus.
4. **Collect into a set** — deduplicate the remaining content lemmas.
5. **Compute delta** — set-difference the two lemma sets to find words
   that were added or removed.

Design principles (from the specification)
------------------------------------------
• **Deterministic**: same input text always produces the same lemma set.
• **Transparent**: every step is inspectable; no hidden inference.
• **No axis attribution**: the pipeline does not know which axis caused
  a word to appear.
• **No embeddings**: operates strictly at the lexical level.
• **No TF-IDF** (Phase 1): results are sorted alphabetically, not by
  corpus rarity.  TF-IDF sorting is reserved for a future phase.

NLTK data requirements
----------------------
This module requires three NLTK data packages:

- ``punkt_tab``  — tokeniser models (Penn Treebank)
- ``stopwords``  — English stopword list (179 words)
- ``wordnet``    — lemmatiser database (WordNet 3.0)

These are downloaded automatically at module load time via
``_ensure_nltk_data()`` with graceful error handling and logging.
Subsequent imports are near-instant because NLTK caches downloaded data
in ``~/nltk_data``.
"""

from __future__ import annotations

import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# NLTK data bootstrap
# -----------------------------------------------------------------------------

# The three NLTK data packages required by this module.  Each entry is a
# tuple of (package_name, nltk.data.find path prefix) so that the bootstrap
# function can locate the correct sub-directory for each package type.
_REQUIRED_NLTK_DATA: tuple[tuple[str, str], ...] = (
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
)


def _ensure_nltk_data() -> None:
    """
    Ensure required NLTK data packages are available locally.

    Downloads any missing packages on first run.  Subsequent imports are
    near-instant because NLTK caches downloaded data in ``~/nltk_data``.

    Logs a warning (not an error) if a download fails so the server can
    still start — the affected functions will raise clear ``LookupError``
    exceptions at call time rather than crashing at import.
    """
    for pkg_name, find_path in _REQUIRED_NLTK_DATA:
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
_ensure_nltk_data()


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Frozen set of English stopwords for O(1) membership testing.
# Loaded once at module level after NLTK data is ensured.
_ENGLISH_STOPWORDS: frozenset[str] = frozenset(stopwords.words("english"))

# Shared lemmatiser instance.  WordNetLemmatizer is stateless and
# thread-safe, so a single instance can be reused across all requests.
_LEMMATIZER: WordNetLemmatizer = WordNetLemmatizer()


# -----------------------------------------------------------------------------
# Private pipeline helpers
# -----------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    """
    Split text into word tokens using NLTK's Penn Treebank tokeniser.

    Returns lowercase tokens that contain at least one alphabetic character.
    Punctuation-only tokens (commas, periods, dashes) and pure numeric
    tokens are discarded because they are structural noise, not content
    signals.

    Parameters
    ----------
    text : str
        Raw input text string.

    Returns
    -------
    list[str]
        Lowercase word tokens, each containing at least one letter.
        Empty list if the input is empty or contains no alphabetic tokens.
    """
    # word_tokenize handles sentence boundaries, contractions, and
    # punctuation splitting according to Penn Treebank conventions.
    raw_tokens = word_tokenize(text)

    # Lowercase and keep only tokens with at least one alpha character.
    # This discards standalone punctuation (".", ",", "--") and pure
    # numbers ("42", "7") while preserving hyphenated words that contain
    # letters and contractions like "n't".
    return [t.lower() for t in raw_tokens if any(c.isalpha() for c in t)]


def _lemmatise(tokens: list[str]) -> list[str]:
    """
    Reduce each token to its base lemma form using the WordNet lemmatiser.

    Uses a two-pass heuristic:

    1. Try **verb** lemmatisation (``pos="v"``).  This catches common
       inflections like "carries" → "carry", "failing" → "fail",
       "walked" → "walk".
    2. If the verb form is unchanged (meaning the word isn't a recognised
       verb inflection), fall back to the default **noun** lemmatisation.
       This handles plurals like "figures" → "figure", "goblins" → "goblin".

    This approach avoids the complexity and additional NLTK data dependency
    of full POS tagging (``averaged_perceptron_tagger``).  For the lab's
    purposes — surfacing lexical pivots in 50–200 word paragraphs — the
    two-pass heuristic is adequate.

    Parameters
    ----------
    tokens : list[str]
        List of lowercase word tokens (output of ``_tokenise``).

    Returns
    -------
    list[str]
        Lemmatised tokens in the same order and of the same length as
        the input.
    """
    result: list[str] = []
    for token in tokens:
        # Pass 1: try verb lemmatisation (catches inflected verbs).
        verb_lemma = _LEMMATIZER.lemmatize(token, pos="v")
        if verb_lemma != token:
            result.append(verb_lemma)
        else:
            # Pass 2: fall back to noun lemmatisation (catches plurals).
            result.append(_LEMMATIZER.lemmatize(token))
    return result


def _filter_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove English stopwords from a token list.

    Stopwords are function words (articles, auxiliaries, pronouns,
    conjunctions) that carry grammatical rather than semantic meaning.
    Filtering them surfaces the content words that actually differ
    between two LLM outputs.

    Uses a ``frozenset`` for O(1) membership testing against the
    ~179 NLTK English stopwords.

    Parameters
    ----------
    tokens : list[str]
        List of lowercase, lemmatised tokens.

    Returns
    -------
    list[str]
        Tokens with all stopwords removed.  Order is preserved.
    """
    return [t for t in tokens if t not in _ENGLISH_STOPWORDS]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def extract_content_lemmas(text: str) -> set[str]:
    """
    Run the full signal isolation pipeline on a text string.

    Pipeline steps (applied in order):

    1. Tokenise — NLTK ``word_tokenize``, lowercase, filter non-alpha.
    2. Lemmatise — WordNet, verb-then-noun fallback.
    3. Filter stopwords — remove NLTK English stopwords.
    4. Collect into a set — deduplicate remaining content lemmas.

    Parameters
    ----------
    text : str
        Raw input text (e.g. an LLM-generated paragraph).

    Returns
    -------
    set[str]
        Unique content lemmas extracted from the text.
        Empty set if the text is empty or contains only stopwords.
    """
    if not text or not text.strip():
        return set()

    tokens = _tokenise(text)
    lemmas = _lemmatise(tokens)
    content = _filter_stopwords(lemmas)
    return set(content)


def compute_delta(
    baseline_text: str,
    current_text: str,
) -> tuple[list[str], list[str]]:
    """
    Compute the content-word delta between two texts.

    Runs the signal isolation pipeline (``extract_content_lemmas``) on
    both texts, then computes set differences to find words that were
    added or removed.

    Parameters
    ----------
    baseline_text : str
        The reference text (A) — typically the stored baseline output.
    current_text : str
        The comparison text (B) — typically the latest generated output.

    Returns
    -------
    tuple[list[str], list[str]]
        A 2-tuple of:

        - **removed** — content lemmas present in A but absent from B,
          sorted alphabetically.
        - **added** — content lemmas present in B but absent from A,
          sorted alphabetically.
    """
    baseline_lemmas = extract_content_lemmas(baseline_text)
    current_lemmas = extract_content_lemmas(current_text)

    removed = sorted(baseline_lemmas - current_lemmas)
    added = sorted(current_lemmas - baseline_lemmas)

    return removed, added
