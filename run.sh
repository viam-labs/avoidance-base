#!/usr/bin/env bash
# Entrypoint for the avoidance-base module.
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck disable=SC1091
source .venv/bin/activate

exec python -m src.main "$@"
