# This file is located in docs/ and generates the pages as long as all needed packages are already installed.

# fail script if subcommand fails
set -e

mkdir -p build

# Generate the example notebooks from their jupytext .py sources
# (only the .py files are versioned, see examples/jupytext.toml).
for f in ../examples/*.py; do
    jupytext --to ipynb "$f" -o "source/$(basename "${f%.py}").ipynb"
done
cp ../examples/ditto.png source/ditto.png

# Concept demos (small, object-focused, as opposed to the module/application
# examples above) live in examples/concepts/ and are prefixed to keep them separate.
for f in ../examples/concepts/*.py; do
    jupytext --to ipynb "$f" -o "source/concepts_$(basename "${f%.py}").ipynb"
done

# Make every webgui scene render as a static preview image that loads the
# interactive 3D only on click (see ipython_startup/00-webgui-static.py),
# instead of every notebook eagerly embedding a full interactive widget --
# that used to make the pages heavy and slow to load.
#
# nbsphinx executes each notebook in a fresh IPython kernel, and IPython
# kernels run every profile_default/startup/*.py from $IPYTHONDIR before any
# cell -- so this patches ngsolve.webgui.Draw without touching the notebooks.
export IPYTHONDIR="$(pwd)/.ipython-build"
mkdir -p "$IPYTHONDIR/profile_default/startup"
cp ipython_startup/*.py "$IPYTHONDIR/profile_default/startup/"
rm -rf webgui_scenes
export WEBGUI_SCENE_DIR="$(pwd)/webgui_scenes"
# WEBGUI_BASE stays unset: the previews then use RELATIVE scene URLs
# (webgui_scenes/<hash>.html), which work locally (file://) as well as under
# a Pages subpath (github.io/ngsxditto/, GitLab Pages /<project>/).
unset WEBGUI_BASE

SPHINX_APIDOC_OPTIONS=members,show-inheritance sphinx-apidoc --templatedir source/_templates/ -o source/ ../ngsxditto
make html

# The static previews link to the full interactive scenes written to
# $WEBGUI_SCENE_DIR during the build; ship them alongside the built pages.
# (build/html/webgui_scenes may already exist from a previous local run --
# without the rm -rf, cp would nest it as build/html/webgui_scenes/webgui_scenes.)
rm -rf build/html/webgui_scenes
cp -r webgui_scenes build/html/webgui_scenes

rm source/*.ipynb
rm source/ditto.png
rm -rf webgui_scenes .ipython-build