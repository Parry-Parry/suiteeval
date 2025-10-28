# Location: suiteeval/pydocs/conf.py
from __future__ import annotations
import os
import sys
from datetime import date

# --- Make the package importable ------------------------------------------------
# pydocs/ is inside suiteeval/, so the project root is two levels up.
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(
    0, os.path.abspath(os.path.join(PROJECT_ROOT, ""))
)  # repo root on sys.path

# --- Project information --------------------------------------------------------
project = "suiteeval"
author = "suiteeval contributors"
copyright = f"{date.today().year}, {author}"
version = ""  # short X.Y
release = ""  # full

# --- General configuration ------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

# Use Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Generate autosummary stub pages automatically
autosummary_imported_members = False
autosummary_generate_overwrite = False
autosummary_generate = False
autosummary_generate_overwrite = False

# Autodoc defaults
autoclass_content = "class"
autodoc_typehints = "description"  # move type hints into the description
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# If some imports are expensive or optional, mock them here.
# autodoc_mock_imports = ["torch", "transformers", "pyterrier", "ir_datasets"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Markdown support via MyST (optional)
myst_enable_extensions = ["colon_fence"]

# Intersphinx: cross-links to Python stdlib
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# --- HTML output ----------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "suiteeval documentation"
