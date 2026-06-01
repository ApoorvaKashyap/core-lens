# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CoreLens"
copyright = "2026, Apoorva Kashyap, Kislaya Shrestha"
author = "Apoorva Kashyap, Kislaya Shrestha"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_copybutton",
    "autoapi.extension",
    "sphinx_multiversion",
]

templates_path = ["_templates"]
exclude_patterns = []


# AutoAPI configuration
autoapi_type = "python"
autoapi_dirs = ["../../src/core_lens"]

# Optional but recommended
autoapi_keep_files = True
autoapi_add_toctree_entry = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST markdown support
myst_enable_extensions = [
    "colon_fence",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Multiversion Configs
smv_branch_whitelist = r"^(main|dev|release-.*)$"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]
