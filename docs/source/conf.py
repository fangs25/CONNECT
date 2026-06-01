"""Sphinx configuration for CONNECT documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "CONNECT"
author = "CONNECT developers"
release = "v0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

autosummary_generate = False
autoclass_content = "class"
autodoc_class_signature = "separated"
templates_path = ["_templates"]
exclude_patterns = []

autodoc_mock_imports = [
    "anndata",
    "hnswlib",
    "numpy",
    "pandas",
    "scanpy",
    "scipy",
    "sklearn",
    "torch",
    "torchvision",
    "tqdm",
]

nbsphinx_execute = "never"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

suppress_warnings = ["autodoc.import_object"]
