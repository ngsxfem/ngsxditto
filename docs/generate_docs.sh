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

SPHINX_APIDOC_OPTIONS=members,show-inheritance sphinx-apidoc --templatedir source/_templates/ -o source/ ../ngsxditto
make html

rm source/*.ipynb
rm source/ditto.png