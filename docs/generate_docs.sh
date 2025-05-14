# This file is located in docs/ and generates the pages as long as all needed packages are already installed.

mkdir -p build

cp ../examples/*.ipynb source/
cp ../examples/ditto.png source/ditto.png

SPHINX_APIDOC_OPTIONS=members,show-inheritance sphinx-apidoc --templatedir source/_templates/ -o source/ ../ngsxditto
make html

rm source/*.ipynb
rm source/ditto.png