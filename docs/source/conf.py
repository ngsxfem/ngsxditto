# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ngsxditto'
copyright = '2025, ditto'
author = 'ditto'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "nbsphinx",
    "jupyter_sphinx",
    "sphinx_mdinclude",
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
]

# make source code visible to sphinx.ext.autodoc
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('../..'))

templates_path = ['_templates']
exclude_patterns = []

master_doc = "contents"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# Quelle: Coverage-HTML (vom pytest --cov-report=html erzeugt)
htmlcov_dir = os.path.abspath(os.path.join("..", "htmlcov"))

# Ziel: in deine Doku (static files)
coverage_target = os.path.join(os.path.dirname(__file__), "_static", "coverage")

if os.path.exists(htmlcov_dir):
    shutil.rmtree(coverage_target, ignore_errors=True)
    shutil.copytree(htmlcov_dir, coverage_target)

    