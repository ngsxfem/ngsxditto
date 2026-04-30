#!/usr/bin/env bash
# Install base Python dependencies required by all CI jobs.
# Call with: bash ci/install_base_deps.sh
set -e

pip install mkl
pip install -r docs/requirements.txt
