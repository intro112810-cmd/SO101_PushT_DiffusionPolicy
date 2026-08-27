#!/usr/bin/env bash
set -euo pipefail
# Generate the four-model comparison JSON/Markdown from anchored evaluations.
# Requires exactly one anchored final evaluation per model.
DF_ROOT=/home/user/kihyun/df
ART=$DF_ROOT/02_InTro_Project/04_experiments/so101_pusht_benchmark
PKG=$DF_ROOT/02_InTro_Project/03_code/so101_pusht_benchmark
P=$DF_ROOT/venvs/paper-baselines/bin/python
INDEX=$ART/artifact-index.json
OUTPUT=${1:-$ART/reports/200ep/comparison}
for model in dp_cnn dp_transformer ibc lstm_gmm; do
  if ! jq -e --arg a "${model//_/-}-200ep-seed-0" '.artifacts[$a].metrics_path != null' "$INDEX" >/dev/null 2>&1; then
    echo "MISSING_EVALUATION $model"
    exit 1
  fi
done
rm -rf "$OUTPUT"
cd "$PKG"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$DF_ROOT/python-cache"
export PYTHONPATH=$PKG/src
export WANDB_MODE=offline
"$P" -m so101_pusht_benchmark.cli compare-models \
  --artifact-index "$INDEX" \
  --artifact-id dp-cnn-200ep-seed-0 \
  --artifact-id dp-transformer-200ep-seed-0 \
  --artifact-id ibc-200ep-seed-0 \
  --artifact-id lstm-gmm-200ep-seed-0 \
  --output "$OUTPUT"
echo "COMPARISON_READY $OUTPUT"
