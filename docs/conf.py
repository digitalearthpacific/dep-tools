# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import subprocess

project = "dep-tools"
copyright = "2025, Digital Earth Pacific"
author = "Jesse Anderson, Alex Leith, Nicholas Metherall"
release = "0.5.4"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    #    "nbsphinx",
    #    "nbsphinx_link",
    #    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# Combine class & class __init__ function docstrings
autoclass_content = "both"
autodoc_member_order = "bysource"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Make files for each module

subprocess.run(
    ["sphinx-apidoc", "-o", "docs/api", "dep_tools", "--separate", "--force"]
)
