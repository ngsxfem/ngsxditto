#!/usr/bin/env bash
# Run the test suite with coverage reports.
# Call with: bash ci/run_tests.sh
#
# Tests marked "slow" (the convergence studies) are skipped by default
# because they dominate the runtime; include them with:
#   RUN_SLOW=1 bash ci/run_tests.sh
set -e

MARKEXPR="not slow"
if [ -n "${RUN_SLOW:-}" ]; then
    MARKEXPR=""
fi

pytest \
    ${MARKEXPR:+-m "$MARKEXPR"} \
    --cov=ngsxditto \
    --cov-report=xml:coverage.xml \
    --cov-report=term-missing \
    --cov-report=html \
    tests/test_*.py
