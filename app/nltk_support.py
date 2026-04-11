"""
Shared NLTK resource management for analysis features.

The Axis Descriptor Lab uses NLTK-backed analysis for signal isolation,
transformation mapping, and micro-indicators. Those features rely on NLTK
data packages that are not guaranteed to exist in a fresh environment.

This module keeps that concern explicit:

- application imports must not perform implicit downloads
- environment preparation may bootstrap the required NLTK data intentionally
- feature code can fail with a clear operator-facing error when resources are
  missing
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import nltk
from nltk.corpus import stopwords

_HOST_NLTK_DATA_DIR = Path("/srv/work/pipeworks/runtime/axis-descriptor-lab/nltk_data")
_LOCAL_NLTK_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "nltk_data"

_BASE_REQUIRED_NLTK_DATA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("punkt_tab", ("tokenizers/punkt_tab", "tokenizers/punkt_tab.zip")),
    ("stopwords", ("corpora/stopwords", "corpora/stopwords.zip")),
    ("wordnet", ("corpora/wordnet", "corpora/wordnet.zip")),
)

_POS_REQUIRED_NLTK_DATA: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "averaged_perceptron_tagger_eng",
        (
            "taggers/averaged_perceptron_tagger_eng",
            "taggers/averaged_perceptron_tagger_eng.zip",
        ),
    ),
)


class NltkResourceError(RuntimeError):
    """Raised when required NLTK data packages are unavailable."""


def _configured_nltk_data_dir() -> Path:
    """Return the explicit NLTK data directory for this environment.

    The Luminal host model should never write NLTK resources into a user home
    directory. Prefer an explicit `NLTK_DATA` override when present, otherwise
    use the host-managed runtime tree when available, and only then fall back
    to a repo-local development path.
    """
    configured = os.getenv("NLTK_DATA")
    if configured:
        first_entry = configured.split(os.pathsep)[0].strip()
        if first_entry:
            return Path(first_entry).expanduser()

    if _HOST_NLTK_DATA_DIR.parent.exists():
        return _HOST_NLTK_DATA_DIR

    return _LOCAL_NLTK_DATA_DIR


def _ensure_search_path(path: Path) -> None:
    """Make sure NLTK searches the configured directory first."""
    resolved = str(path)
    if resolved not in nltk.data.path:
        nltk.data.path.insert(0, resolved)


def _required_packages(*, require_pos_tagger: bool) -> tuple[tuple[str, str], ...]:
    """Return the required NLTK package specs for one analysis context."""
    if require_pos_tagger:
        return _BASE_REQUIRED_NLTK_DATA + _POS_REQUIRED_NLTK_DATA
    return _BASE_REQUIRED_NLTK_DATA


def _missing_packages(*, require_pos_tagger: bool) -> list[str]:
    """Return the names of missing required NLTK data packages."""
    _ensure_search_path(_configured_nltk_data_dir())
    missing: list[str] = []
    for pkg_name, find_paths in _required_packages(require_pos_tagger=require_pos_tagger):
        for find_path in find_paths:
            try:
                nltk.data.find(find_path)
                break
            except LookupError:
                continue
        else:
            missing.append(pkg_name)
    return missing


def _missing_message(missing: list[str]) -> str:
    """Build a clear operator-facing message for missing NLTK data."""
    packages = ", ".join(sorted(missing))
    return (
        "Required NLTK data packages are missing: "
        f"{packages}. "
        "Bootstrap them explicitly with "
        "`python tools/bootstrap_nltk.py` inside the repo venv before using "
        "analysis features."
    )


def ensure_nltk_data(*, require_pos_tagger: bool = False) -> None:
    """Validate that required NLTK data packages are available locally."""
    missing = _missing_packages(require_pos_tagger=require_pos_tagger)
    if missing:
        raise NltkResourceError(_missing_message(missing))


def bootstrap_nltk_data(*, require_pos_tagger: bool = True, quiet: bool = False) -> None:
    """Download the required NLTK data packages explicitly for one environment."""
    download_dir = _configured_nltk_data_dir()
    download_dir.mkdir(parents=True, exist_ok=True)
    _ensure_search_path(download_dir)

    for pkg_name, _ in _required_packages(require_pos_tagger=require_pos_tagger):
        nltk.download(pkg_name, download_dir=str(download_dir), quiet=quiet)

    missing = _missing_packages(require_pos_tagger=require_pos_tagger)
    if missing:
        raise NltkResourceError(_missing_message(missing))


@lru_cache(maxsize=1)
def english_stopwords() -> frozenset[str]:
    """Return the English stopword set after validating resource availability."""
    ensure_nltk_data()
    return frozenset(stopwords.words("english"))
