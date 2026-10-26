#!/usr/bin/env bash
# Entrypoint for the avoidance-base module.
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck disable=SC1091
source .venv/bin/activate

# Cloud transforms are small matmuls. Leaving OpenBLAS / MKL free to spawn
# a thread per core turns two 10 Hz cameras into multi-core pegging.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

exec python -m src.main "$@"
