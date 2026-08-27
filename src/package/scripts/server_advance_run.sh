#!/usr/bin/env bash
set -euo pipefail
# After MODEL seed-SEED training completes: export bundle, start next seed,
# then evaluate every bundle with a metrics_path-free record.
# Invocation: server_advance_run.sh MODEL SEED
DF_ROOT=/home/user/kihyun/df
ART=$DF_ROOT/02_InTro_Project/04_experiments/so101_pusht_benchmark
PKG=$DF_ROOT/02_InTro_Project/03_code/so101_pusht_benchmark
MODEL=$1
SEED=$2
RECEIPT="$ART/models/200ep/$MODEL/seed-$SEED/full/training_receipt.json"
if ! test -f "$RECEIPT"; then
  echo "NO_RECEIPT $MODEL seed-$SEED"
  exit 1
fi
# 1. bundle export
bash "$PKG/scripts/server_export_and_next.sh" "$MODEL" "$SEED"
# 2. rolling wave to next seed (only if this seed's run is complete)
bash "$PKG/scripts/server_rolling_wave.sh" "$MODEL" || true
# 3. evaluate all bundles that lack metrics
bash "$PKG/scripts/server_evaluate_all.sh"
echo "ADVANCED $MODEL seed-$SEED"
