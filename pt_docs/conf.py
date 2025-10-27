# -- Path setup --------------------------------------------------------------
import os, sys
# If your package uses a src/ layout, point to it:
sys.path.insert(0, os.path.abspath('../src'))
# If not, and your package is at repo root as a top-level module:
# sys.path.insert(0, os.path.abspath('..'))

# -- Project info ------------------------------------------------------------
project = "suiteeval"
author = "Andrew Parry"

# -- General config ----------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # handles Google/NumPy docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.duration",
    "sphinx_autodoc_typehints",  # renders type hints in the docs
]

autosummary_generate = True      # generate pages for autosummary directives
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__call__",
    "inherited-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"  # or "furo", "pydata_sphinx_theme", etc.
