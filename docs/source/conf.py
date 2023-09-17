# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import datetime
from sphinx.ext import autodoc

sys.path.insert(0, os.path.abspath('..'))

try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GPmp'
current_year = datetime.date.today().year
if current_year > 2022:
    copyright = f'2022-{current_year}, CentraleSupelec'
else:
    copyright = f'2022, CentraleSupelec'
author = 'Emmanuel Vazquez'
# https://stackoverflow.com/questions/26141851/let-sphinx-use-version-from-setup-py
release = metadata.version('gpmp')
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


# -- Extensions -------------------------------------------------------------

autosummary_generate = True
numpydoc_class_members_toctree = False
imgmath_latex_preamble = r'\usepackage[utf8]{inputenc}'


# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme_options = {
    "logo": "logo.png",
    "logo_name": True,
    "description": "the GP micro package",
    "github_user": "gpmp-dev",
    "github_repo": "gpmp",
    "github_banner": True,
    "github_button": False,
    "travis_button": False,
    "codecov_button": False,
    "analytics_id": False,  # TODO
    "font_family": "'Roboto', Georgia, sans",
    "head_font_family": "'Roboto', Georgia, serif",
    "code_font_family": "'Roboto Mono', 'Consolas', monospace",
    "pre_bg": "#433e56",
}
