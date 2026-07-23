#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

VENV="${ROOT}/.venv"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install -r "${ROOT}/requirements.txt"
  "$VENV/bin/pip" install "opencv-python>=4.8.0"
fi

# default to live if no args
if [[ $# -eq 0 ]]; then
  set -- live
fi

exec "$VENV/bin/python" -m manual_tracking "$@"
