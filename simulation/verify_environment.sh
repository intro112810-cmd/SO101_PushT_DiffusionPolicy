#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s EVIDENCE_DIR\n' "$0" >&2
  exit 2
fi

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="$PACKAGE_ROOT/src/package"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$SOURCE_ROOT"
MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" \
  -m so101_pusht_benchmark.cli native-env-smoke \
  --seed 100000 --steps 1 --evidence "$1"
