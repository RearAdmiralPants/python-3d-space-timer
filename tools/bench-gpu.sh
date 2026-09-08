#!/usr/bin/env bash
#
# Benchmark sky.frag on each GPU this machine has, at the display's resolution.
#
# Answers the only question that matters on a hybrid-graphics laptop: is the
# discrete GPU actually worth the PRIME offload? On some pairings (a modern
# Iris Xe against an old Quadro) it is not, and the answer is not guessable.
#
#   tools/bench-gpu.sh [WIDTH HEIGHT [FRAMES]]

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./gpu-select.sh

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
    echo "error: no .venv -- run ./launch.sh once to build it." >&2
    exit 1
fi

for gpu in intel nvidia; do
    echo "=============== $gpu ==============="
    if [ "$gpu" = nvidia ] && ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "(skipped: no NVIDIA driver on this machine)"
        echo
        continue
    fi
    # Subshell: each iteration gets a clean copy of the environment, so the
    # offload variables from one pass cannot leak into the next.
    ( gpu_select "$gpu" >/dev/null; PYTHONPATH=src exec "$PY" tools/bench_sky.py "$@" )
    echo
done
