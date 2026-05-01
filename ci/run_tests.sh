#!/usr/bin/env bash
# Run the test suite with coverage reports.
# Call with: bash ci/run_tests.sh
set -e

pytest \
    --cov=ngsxditto \
    --cov-report=xml:coverage.xml \
    --cov-report=term-missing \
    --cov-report=html \
    tests/test_*.py
