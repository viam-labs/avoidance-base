#!/usr/bin/env bash
# First-run setup: create a virtualenv and install module requirements.
set -euo pipefail

cd "$(dirname "$0")"

VENV_NAME=".venv"
PYTHON=${PYTHON:-python3}

log() { echo "[avoidance-base setup] $*"; }
die() { echo "[avoidance-base setup] ERROR: $*" >&2; exit 1; }

if ! "$PYTHON" --version >/dev/null 2>&1; then
    die "python3 not found on PATH"
fi
if [ ! -d "${VENV_NAME}" ]; then
    log "creating virtualenv in ${VENV_NAME}"
    if ! "$PYTHON" -m venv --system-site-packages "${VENV_NAME}"; then
        die "failed to create ${VENV_NAME}; install python3-venv and re-run setup"
    fi
fi
# shellcheck disable=SC1091
source "${VENV_NAME}/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt -q
log "setup complete"
