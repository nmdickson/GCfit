# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'GCfit'
copyright = '2024, Nolan Dickson'
author = 'Nolan Dickson'

# The full version, including alpha/beta/rc tags
release = '1.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx_toggleprompt',
    'numpydoc'
]

autosummary_generate = True

autodoc_inherit_docstrings = False

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
# numpydoc_class_members_toctree = True
# numpydoc_validation_checks = {"all", "GL01", "EX01", "SA01"}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [{
        "name": "GitHub",
        "url": "https://github.com/nmdickson/GCfit",
        "icon": "fab fa-github-square",
        "type": "fontawesome",  # Default is fontawesome
    }]
}

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise interprise
    "github_user": "nmdickson",
    "github_repo": "GCfit",
    "github_version": "develop",
    "doc_path": "docs/source",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
