#!/usr/bin/env bash
set -euo pipefail
# After a model's training_receipt.json appears, export its inference bundle.
# Invocation: server_export_and_next.sh MODEL
# MODEL: dp_cnn | dp_transformer | ibc | lstm_gmm
DF_ROOT=/home/user/kihyun/df
ART=$DF_ROOT/02_InTro_Project/04_experiments/so101_pusht_benchmark
PKG=$DF_ROOT/02_InTro_Project/03_code/so101_pusht_benchmark
P=$DF_ROOT/venvs/paper-baselines/bin/python
INDEX=$ART/artifact-index.json
MODEL=$1
SEED=$2
RECEIPT="$ART/models/200ep/$MODEL/seed-$SEED/full/training_receipt.json"
if ! test -f "$RECEIPT"; then
  echo "NO_RECEIPT $MODEL seed-$SEED"
  exit 1
fi
CHECKPOINT="$ART/models/200ep/$MODEL/seed-$SEED/full/checkpoints/latest.ckpt"
CONFIG="$ART/models/200ep/$MODEL/seed-$SEED/full/resolved_config.json"
OUTPUT="$ART/models/200ep/$MODEL/seed-$SEED/bundle"
ARTIFACT=$(printf "%s-200ep-seed-%s" "$MODEL" "$SEED" | tr _ -)
rm -rf "$OUTPUT"
cd "$PKG"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$DF_ROOT/python-cache"
export PYTHONPATH=$PKG/src
export WANDB_MODE=offline
"$P" -m so101_pusht_benchmark.cli export-inference-bundle \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  --artifact-id "$ARTIFACT" \
  --artifact-index "$INDEX"
echo "EXPORTED $MODEL seed-$SEED -> $OUTPUT ($ARTIFACT)"
