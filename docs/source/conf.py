#!/usr/bin/env python3
# -- Path setup --------------------------------------------------------------

from pathlib import Path
import sys

import binney
base_dir = Path(binney.__file__).parent

about = {}
with (base_dir / "__about__.py").open() as f:
    exec(f.read(), about)

sys.path.insert(0, Path('..').resolve())


# -- Project information -----------------------------------------------------

project = about['__title__']
copyright = f'2020, {about["__author__"]}'
author = about["__author__"]

# The short X.Y version.
version = about["__version__"]
# The full version, including alpha/beta/rc tags.
release = about["__version__"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

needs_sphinx = '1.5'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_click.ext',
    'sphinx_autodoc_typehints',
    'matplotlib.sphinxext.plot_directive',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = '.rst'
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
autodoc_mock_imports = ['ipopt']

add_module_names = False


def setup(app):
    app.add_css_file("theme.css")
