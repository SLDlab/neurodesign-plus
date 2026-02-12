"""Sphinx configuration for neurodesign-plus documentation."""

import importlib.metadata

# -- Project information -----------------------------------------------------
project = "neurodesign-plus"
copyright = "2016-2026, Joke Durnez, Atharv Umap, Valentin Guigon"
author = "Joke Durnez, Atharv Umap, Valentin Guigon"

# Version from installed package metadata
try:
    release = importlib.metadata.version("neurodesign-plus")
except importlib.metadata.PackageNotFoundError:
    release = "dev"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# MyST-parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "neurodesign-plus"

html_theme_options = {
    "source_repository": "https://github.com/SLDlab/neurodesign-plus",
    "source_branch": "master",
    "source_directory": "docs/",
}

html_static_path = ["_static"]
templates_path = []

# -- Options for LaTeX output ------------------------------------------------
latex_documents = [
    (
        "index",
        "neurodesign-plus.tex",
        "neurodesign-plus Documentation",
        "Joke Durnez, Atharv Umap, Valentin Guigon",
        "manual",
    ),
]
