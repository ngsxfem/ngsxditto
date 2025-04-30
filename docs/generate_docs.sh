# This file is located in docs/ and generates the pages as long as all needed packages are already installed.

mkdir -p build

cp ../examples/ditto_lset.ipynb source/ditto_lset.ipynb
cp ../examples/ditto.png source/ditto.png

make html

rm source/ditto_lset.ipynb
rm source/ditto.png