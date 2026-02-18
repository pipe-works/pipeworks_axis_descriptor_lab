# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Add the project root to the path so Sphinx can find the app package
sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -----------------------------------------------------

project = "Axis Descriptor Lab"
copyright = "2026, aapark"
author = "aapark"

# Read version from pyproject.toml (single source of truth)
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_pyproject = Path(__file__).parent.parent / "pyproject.toml"
with open(_pyproject, "rb") as f:
    _data = tomllib.load(f)
release = _data["project"]["version"]
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate documentation from docstrings
    "sphinx.ext.napoleon",  # Support for Google-style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.intersphinx",  # Link to other project documentation
    "sphinx_autodoc_typehints",  # Better type hint rendering
    "myst_parser",  # Markdown support
]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"

# -- MyST-Parser configuration -----------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # ::: admonition syntax
    "deflist",  # definition lists
    "fieldlist",  # field lists
]
myst_heading_anchors = 3

# -- HTML output options -----------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
}

# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# -- Napoleon settings (Google-style docstrings) ----------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
