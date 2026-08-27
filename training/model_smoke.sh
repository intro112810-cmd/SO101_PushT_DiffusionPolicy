#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  printf 'usage: %s MODEL [model-smoke options]\n' "$0" >&2
  exit 2
fi

MODEL="$1"
shift

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="$PACKAGE_ROOT/src/package"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$SOURCE_ROOT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" \
  -m so101_pusht_benchmark.cli model-smoke --model "$MODEL" "$@"
