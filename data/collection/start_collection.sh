#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="$PACKAGE_ROOT/src/package"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$SOURCE_ROOT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" \
  -m so101_pusht_benchmark.cli collect-native --launch "$@"
