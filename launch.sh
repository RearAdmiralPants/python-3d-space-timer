#!/usr/bin/env bash
#
# Dev launcher: creates .venv on first run, then starts the timer.
#
# Re-installs dependencies whenever pyproject.toml changes, so adding a
# dependency doesn't silently leave you on a stale environment.
#
# Any remaining arguments are forwarded to the app. Set NAIVE_TIMER_TUNE=0 to
# launch without the live shader-tuning panel.
#
#   ./launch.sh                  # whatever GPU the system picks (usually the iGPU)
#   ./launch.sh --gpu nvidia     # force the discrete GPU via PRIME offload
#   ./launch.sh --gpu intel      # force the integrated GPU
#   ./launch.sh --gpu list       # what GPUs does this machine have?

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

source ./gpu-select.sh
gpu_parse_args "$@"
shift "$GPU_SHIFT"

VENV=.venv
PY="$VENV/bin/python"
STAMP="$VENV/.pyproject.sha256"

if [ ! -x "$PY" ]; then
    if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
        echo "error: python3 is missing the 'ensurepip' module, so it cannot create a venv." >&2
        echo "On Debian/Ubuntu, install it with:" >&2
        echo >&2
        echo "    sudo apt install python3-venv" >&2
        echo >&2
        exit 1
    fi

    echo ">> creating virtualenv in $VENV"
    rm -rf "$VENV"
    python3 -m venv "$VENV"
fi

# Install (or re-install) the project whenever pyproject.toml has changed.
want=$(sha256sum pyproject.toml | cut -d' ' -f1)
have=$(cat "$STAMP" 2>/dev/null || true)

if [ "$want" != "$have" ]; then
    echo ">> installing dependencies (first run pulls ~250 MB of Qt, so give it a minute)"
    "$PY" -m pip install --upgrade --quiet pip
    "$PY" -m pip install --editable .
    printf '%s\n' "$want" >"$STAMP"
fi

gpu_select "$GPU_CHOICE"

export NAIVE_TIMER_TUNE="${NAIVE_TIMER_TUNE:-1}"
exec "$PY" -m naive_timer "$@"
