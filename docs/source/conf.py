# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import datetime

_DOCS_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DOCS_SOURCE_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("GPMP_LOG_LEVEL", "WARNING")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GPmp'
current_year = datetime.date.today().year
if current_year > 2022:
    copyright = f'2022-{current_year}, CentraleSupelec'
else:
    copyright = f'2022, CentraleSupelec'
author = 'Emmanuel Vazquez'
with open(os.path.join(_REPO_ROOT, "VERSION"), encoding="utf-8") as f:
    release = f.read().strip()
version = release
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'sphinx.ext.intersphinx',
    "sphinx.ext.napoleon",
    "numpydoc",
    'sphinx.ext.viewcode',
    "jupyter_sphinx",
    "sphinx.ext.mathjax"
]

USE_IMGMATH = False
if USE_IMGMATH:
    try:
        import sphinx.ext.imgmath  # noqa
    except ImportError:
        extensions.append('sphinx.ext.pngmath')
    else:
        extensions.append('sphinx.ext.imgmath')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["images"]

# The name of the Pygments (syntax highlighting) style to use.
# https://stackoverflow.com/questions/48615629/how-to-include-pygments-styles-in-a-sphinx-project
# https://stylishthemes.github.io/Syntax-Themes/pygments/
# https://github.com/theacodes/witchhazel
pygments_style = "witchhazel.WitchHazelStyle"
pygments_dark_style = "witchhazel.WitchHazelStyle"


# -- Extensions -------------------------------------------------------------

autosummary_generate = True
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
imgmath_latex_preamble = r'\usepackage[utf8]{inputenc}'


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_logo = "_static/logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme_options = {
    "source_repository": "https://github.com/gpmp-dev/gpmp/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#003d45",
        "color-brand-content": "#027bab",
        "color-api-name": "#003d45",
        "color-api-pre-name": "#027bab",
    },
    "dark_css_variables": {
        "color-brand-primary": "#80d8e6",
        "color-brand-content": "#80d8e6",
        "color-api-name": "#80d8e6",
        "color-api-pre-name": "#9ca0a5",
    },
}
