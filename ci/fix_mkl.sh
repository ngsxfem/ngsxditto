#!/usr/bin/env bash
# Fix MKL soname mismatch: ngsolve links against libmkl_rt.so.2 but
# mkl 2026+ ships libmkl_rt.so.3. Creates a compatibility symlink.
#
# Usage:
#   bash ci/fix_mkl.sh      -- GitHub Actions (writes to $GITHUB_ENV)
#   source ci/fix_mkl.sh    -- GitLab CI (exports LD_LIBRARY_PATH in current shell)

PYTHON_PREFIX=$(python3 -c "import sys; print(sys.prefix)")
MKL_SO3=$(find "$PYTHON_PREFIX" -name "libmkl_rt.so.3" 2>/dev/null | head -1)

if [ -n "$MKL_SO3" ]; then
    mkdir -p /tmp/mkl_compat
    ln -sf "$MKL_SO3" /tmp/mkl_compat/libmkl_rt.so.2
    export LD_LIBRARY_PATH=/tmp/mkl_compat:${LD_LIBRARY_PATH}
    # Persist to subsequent steps in GitHub Actions
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "LD_LIBRARY_PATH=/tmp/mkl_compat:${LD_LIBRARY_PATH}" >> "$GITHUB_ENV"
    fi
fi
