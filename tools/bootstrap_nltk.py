"""
Explicit NLTK data bootstrap for Axis Descriptor Lab environments.

Run this inside the repo's active virtual environment to install the NLTK
resources required by the analysis features. This keeps environment
preparation explicit and avoids hidden downloads during application import.
"""

from __future__ import annotations

from app.nltk_support import bootstrap_nltk_data


def main() -> None:
    """Download the NLTK data packages required by analysis features."""
    bootstrap_nltk_data(require_pos_tagger=True, quiet=False)
    print("Axis Descriptor Lab NLTK data bootstrap complete.")


if __name__ == "__main__":
    main()
