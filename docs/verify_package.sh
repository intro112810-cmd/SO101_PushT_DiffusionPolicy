#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAS_ARTIFACT_ROOT="${NAS_ARTIFACT_ROOT:-$PACKAGE_ROOT/../NAS_Artifacts/InTro_SO101_PushT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

required_paths=(
  "README.md"
  "docs/FINAL_HANDOFF_CHECKLIST.md"
  "docs/verify_final_handoff.py"
  "src/package/src/so101_pusht_benchmark"
  "data/datasets/frozen_split_200ep/splits.json"
  "evaluation/four_model_comparison.csv"
  "sim_to_real/evidence/EVIDENCE_INDEX.json"
  "sim_to_real/evidence/shadow/terminal_receipt.json"
  "sim_to_real/evidence/shadow/ledger.jsonl"
  "sim_to_real/final_results/FINAL_STATUS.json"
  "docs/SIM_TO_REAL_FINAL_2026-08-27.md"
  "integrity/FINAL_HANDOFF_MANIFEST.tsv"
  "integrity/MANIFEST.tsv"
  "integrity/SHA256SUMS_GIT"
  "integrity/SHA256SUMS_NAS"
)
for relative_path in "${required_paths[@]}"; do
  test -e "$PACKAGE_ROOT/$relative_path"
done

required_nas_paths=(
  "datasets/native_store_200ep/manifest.json"
  "models/dp_cnn/training/final_checkpoint.ckpt"
  "models/dp_cnn/bundle/policy.safetensors"
  "models/dp_transformer/training/final_checkpoint.ckpt"
  "models/dp_transformer/bundle/policy.safetensors"
  "models/ibc/training/final_checkpoint.ckpt"
  "models/ibc/bundle/policy.safetensors"
  "models/lstm_gmm/training/final_checkpoint.ckpt"
  "models/lstm_gmm/bundle/policy.safetensors"
  "runtime/pushT-so100.git/HEAD"
)
for relative_path in "${required_nas_paths[@]}"; do
  test -e "$NAS_ARTIFACT_ROOT/$relative_path"
done

"$PYTHON_BIN" "$PACKAGE_ROOT/docs/verify_final_handoff.py"
(
  cd "$PACKAGE_ROOT"
  sha256sum --check --quiet "integrity/SHA256SUMS_GIT"
)
(
  cd "$NAS_ARTIFACT_ROOT"
  sha256sum --check --quiet "$PACKAGE_ROOT/integrity/SHA256SUMS_NAS"
)
printf 'PACKAGE_OK
'
