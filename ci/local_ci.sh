#!/usr/bin/env bash
# Run the CI stages (see .gitlab-ci.yml / .github/workflows/ci.yml) locally.
#
# The working-tree state of all *tracked* files is synced into a separate
# work directory (so this reproduces "what CI would do if I committed and
# pushed my current changes" without touching the checkout), and each stage
# runs there in its own virtualenv -- mirroring the per-job isolation of CI.
#
# Usage:
#   bash ci/local_ci.sh [options] <stage> [<stage> ...]
#
# Stages:
#   build      install deps + package, verify import        (job: build_src)
#   test       pytest with coverage reports                 (job: test_src)
#   pages      build the sphinx/jupytext documentation      (job: build_pages)
#   package    build the sdist/wheel                        (job: build_package)
#   all        build test pages package
#   clean      remove the work directory
#
# Options:
#   --docker         run the stage inside the python:3.12 image (same as CI).
#                    Recommended: exactly reproduces the CI environment.
#   --workdir DIR    work directory (default: <repo>/../ngsxditto-ci,
#                    or $NGSXDITTO_CI_DIR if set)
#   --fresh          delete the stage's virtualenv first (fully fresh, as CI)
#
# Extra pytest arguments can be passed via PYTEST_ADDOPTS, e.g. to profile
# slow tests or skip the ones marked slow:
#   PYTEST_ADDOPTS='--durations=25' bash ci/local_ci.sh test
#   PYTEST_ADDOPTS='-m "not slow"'  bash ci/local_ci.sh --docker test
#
# Outputs land in the work directory:
#   src/coverage.xml, src/htmlcov/   (test)
#   public/                          (pages -- same as the deployed pages)
#   dist/                            (package)

set -euo pipefail

DOCKER_IMAGE=python:3.12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
die() { echo "local_ci: error: $*" >&2; exit 1; }
note() { echo -e "\n=== local_ci: $* ===\n"; }

repo_root() { git rev-parse --show-toplevel; }

sync_src() {
    local repo=$1 dst=$2
    mkdir -p "$dst"
    # Tracked files as they are in the working tree, minus deleted ones,
    # plus anything not-yet-committed inside ci/ (so this script itself
    # works before it is committed).
    { git -C "$repo" ls-files
      git -C "$repo" ls-files -o --exclude-standard ci/
    } | sort -u > "$dst/.filelist"
    ( cd "$repo" && rsync -a --delete --delete-excluded \
        --files-from="$dst/.filelist" --ignore-missing-args \
        ./ "$dst/src/" )
}

make_venv() {
    local venv=$1
    if [ -n "${FRESH:-}" ] && [ -d "$venv" ]; then
        note "removing $venv (--fresh)"
        rm -rf "$venv"
    fi
    if [ ! -d "$venv" ]; then
        note "creating virtualenv $venv with $PYTHON_BIN"
        "$PYTHON_BIN" -m venv "$venv"
    fi
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    # CI runs with a clean environment -- a developer shell often points
    # PYTHONPATH / LD_LIBRARY_PATH / NETGENDIR at a local ngsolve build,
    # which must not leak into the stage.
    unset PYTHONPATH PYTHONHOME NETGENDIR
    export LD_LIBRARY_PATH=""   # fix_mkl.sh appends to it
}

# ---------------------------------------------------------------------------
# Stages -- each mirrors the corresponding CI job line by line.
# ---------------------------------------------------------------------------
stage_build() {        # .gitlab-ci.yml: build_src
    make_venv "$WORKDIR/$VENV_PREFIX-build"
    bash ci/install_base_deps.sh
    pip install pyvista
    pip install .
    source ci/fix_mkl.sh
    python3 -c "import ngsxditto"
    note "build stage OK (import succeeded)"
}

stage_test() {         # .gitlab-ci.yml: test_src
    make_venv "$WORKDIR/$VENV_PREFIX-test"
    pip install pytest pytest-cov
    bash ci/install_base_deps.sh
    pip install -e .
    source ci/fix_mkl.sh
    bash ci/run_tests.sh
    note "test stage OK -- coverage.xml and htmlcov/ in $PWD"
}

stage_pages() {        # .gitlab-ci.yml: build_pages
    if [ -n "${LOCAL_CI_IN_CONTAINER:-}" ]; then
        apt-get update
        apt-get install -y libosmesa6-dev libgl1 mesa-utils make pandoc
    else
        command -v pandoc >/dev/null || die "pages stage needs pandoc (or use --docker)"
        command -v make   >/dev/null || die "pages stage needs make (or use --docker)"
    fi
    make_venv "$WORKDIR/$VENV_PREFIX-pages"
    pip install --upgrade pip setuptools wheel
    bash ci/install_base_deps.sh
    pip install .
    source ci/fix_mkl.sh
    (
        cd docs
        mkdir -p htmlcov
        cp -r ../htmlcov/* htmlcov/ 2>/dev/null || true
        export PYVISTA_OFF_SCREEN=true
        export VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=true
        export DISPLAY=:99.0
        ./generate_docs.sh
    )
    rm -rf "$WORKDIR/public"
    mv docs/build/html "$WORKDIR/public"
    note "pages stage OK -- open $WORKDIR/public/index.html"
}

stage_package() {      # .gitlab-ci.yml: build_package
    make_venv "$WORKDIR/$VENV_PREFIX-package"
    pip install .
    pip install build twine
    python -m build
    rm -rf "$WORKDIR/dist"
    mv dist "$WORKDIR/dist"
    note "package stage OK -- artifacts in $WORKDIR/dist"
}

run_stage() {
    case $1 in
        build)   stage_build ;;
        test)    stage_test ;;
        pages)   stage_pages ;;
        package) stage_package ;;
        *) die "unknown stage: $1" ;;
    esac
}

# ---------------------------------------------------------------------------
# In-container entry point: the host invocation re-executes this script
# inside the docker image with LOCAL_CI_IN_CONTAINER / LOCAL_CI_STAGE set.
# ---------------------------------------------------------------------------
if [ -n "${LOCAL_CI_IN_CONTAINER:-}" ]; then
    cd /ci/src
    WORKDIR=/ci
    VENV_PREFIX=venv-docker
    PYTHON_BIN=python3
    export PIP_CACHE_DIR=/ci/pip-cache
    trap 'chown -R "${LOCAL_CI_UIDGID}" /ci' EXIT
    run_stage "$LOCAL_CI_STAGE"
    exit 0
fi

# ---------------------------------------------------------------------------
# Host-side argument parsing and dispatch
# ---------------------------------------------------------------------------
REPO=$(repo_root)
WORKDIR=${NGSXDITTO_CI_DIR:-$(dirname "$REPO")/ngsxditto-ci}
USE_DOCKER=
FRESH=
STAGES=()

while [ $# -gt 0 ]; do
    case $1 in
        --docker)  USE_DOCKER=1 ;;
        --fresh)   FRESH=1 ;;
        --workdir) [ $# -ge 2 ] || die "--workdir needs an argument"; WORKDIR=$2; shift ;;
        --workdir=*) WORKDIR=${1#*=} ;;
        -h|--help) awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/,""); print}' "$0"; exit 0 ;;
        all)       STAGES+=(build test pages package) ;;
        clean)     STAGES+=(clean) ;;
        build|test|pages|package) STAGES+=("$1") ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
    shift
done

[ ${#STAGES[@]} -gt 0 ] || die "no stage given (see --help)"
WORKDIR=$(mkdir -p "$WORKDIR" && cd "$WORKDIR" && pwd)

for stage in "${STAGES[@]}"; do
    if [ "$stage" = clean ]; then
        note "removing $WORKDIR"
        rm -rf "$WORKDIR"
        continue
    fi

    note "syncing tracked files -> $WORKDIR/src"
    sync_src "$REPO" "$WORKDIR"

    if [ -n "$USE_DOCKER" ]; then
        note "stage '$stage' in docker ($DOCKER_IMAGE)"
        docker run --rm -t \
            -v "$WORKDIR":/ci \
            -e LOCAL_CI_IN_CONTAINER=1 \
            -e LOCAL_CI_STAGE="$stage" \
            -e LOCAL_CI_UIDGID="$(id -u):$(id -g)" \
            -e PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-}" \
            -w /ci/src \
            "$DOCKER_IMAGE" bash ci/local_ci.sh
    else
        note "stage '$stage' natively"
        # CI uses python 3.12; prefer a real 3.12 (system, else uv-managed).
        if command -v python3.12 >/dev/null; then
            PYTHON_BIN=python3.12
        elif command -v uv >/dev/null; then
            note "no system python3.12 -- using uv-managed CPython 3.12"
            uv python install 3.12 --quiet
            PYTHON_BIN=$(uv python find 3.12)
        else
            PYTHON_BIN=python3
            ver=$($PYTHON_BIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')
            [ "$ver" = "3.12" ] || echo "local_ci: warning: CI uses python 3.12," \
                "you have $ver -- results may differ (consider --docker)" >&2
        fi
        VENV_PREFIX=venv-native
        export PIP_CACHE_DIR="$WORKDIR/pip-cache"
        ( cd "$WORKDIR/src" && run_stage "$stage" )
    fi
done

note "done: ${STAGES[*]}"
