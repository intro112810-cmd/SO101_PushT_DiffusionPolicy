#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  printf 'usage: %s MODEL OUTPUT_DIR [evaluate-model options]\n' "$0" >&2
  exit 2
fi

MODEL="$1"
OUTPUT_DIR="$2"
shift 2

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="$PACKAGE_ROOT/src/package"
NAS_ARTIFACT_ROOT="${NAS_ARTIFACT_ROOT:-$PACKAGE_ROOT/../NAS_Artifacts/InTro_SO101_PushT}"
BUNDLE_ROOT="$NAS_ARTIFACT_ROOT/models/$MODEL/bundle"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$SOURCE_ROOT"
MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" \
  -m so101_pusht_benchmark.cli evaluate-model \
  --model "$MODEL" --bundle "$BUNDLE_ROOT" --output "$OUTPUT_DIR" "$@"
