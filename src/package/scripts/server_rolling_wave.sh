#!/usr/bin/env bash
set -euo pipefail
# Rolling wave: start the next unfinished seed (1 or 2) of a completed model.
# Invocation: server_rolling_wave.sh MODEL
# MODEL: dp_cnn | dp_transformer | ibc | lstm_gmm
DF_ROOT=/home/user/kihyun/df
ART=$DF_ROOT/02_InTro_Project/04_experiments/so101_pusht_benchmark
PKG=$DF_ROOT/02_InTro_Project/03_code/so101_pusht_benchmark
PROF=$PKG/configs/experiment/pusht_so100_paper_faithful_200ep_v1.yaml
MODEL=$1
case $MODEL in
  dp_cnn | dp_transformer) UPDATES=1794000 ;;
  ibc) UPDATES=100000 ;;
  lstm_gmm) UPDATES=300000 ;;
  *) echo "bad-model: $MODEL" >&2; exit 2 ;;
esac
for seed in 1 2; do
  unit="kihyun-pusht-${MODEL//_/-}-seed-${seed}-train"
  if systemctl --user is-active --quiet "$unit.service"; then
    echo "ALREADY_ACTIVE $unit"
    exit 0
  fi
  if test -e "$ART/models/200ep/$MODEL/seed-$seed/full/training_receipt.json"; then
    echo "ALREADY_COMPLETE $unit"
    exit 0
  fi
  prev=$((seed - 1))
  if ! test -e "$ART/models/200ep/$MODEL/seed-$prev/full/training_receipt.json"; then
    echo "PREREQUISITE_MISSING $MODEL seed-$prev"
    exit 1
  fi
  rm -rf "$ART/models/200ep/$MODEL/seed-$seed/full"
  rm -f "$DF_ROOT/resource-logs/$unit.jsonl" "$DF_ROOT/service-receipts/$unit.json"
  systemctl --user reset-failed "$unit.service" 2>/dev/null || true
  artifact=$(printf "%s-200ep-seed-%s" "$MODEL" "$seed" | tr _ -)
  bash "$PKG/scripts/server_systemd_training.sh" start \
    "$MODEL" "$seed" \
    "$ART/datasets/frozen_four_model_200ep" \
    "$ART/models/200ep/$MODEL/seed-$seed/full" \
    "$artifact" "$ART/artifact-index.json" "$PROF" "$UPDATES"
  echo "STARTED $unit"
  exit 0
done
echo "NO_NEXT_SEED $MODEL"
