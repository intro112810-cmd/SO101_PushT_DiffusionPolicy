#!/usr/bin/env bash
set -euo pipefail
# Evaluate every 200ep run whose bundle exists, on ordered seeds 100000..100099.
# Idempotent: skips runs whose evaluation is already anchored in the index.
DF_ROOT=/home/user/kihyun/df
ART=$DF_ROOT/02_InTro_Project/04_experiments/so101_pusht_benchmark
PKG=$DF_ROOT/02_InTro_Project/03_code/so101_pusht_benchmark
NATIVE=$DF_ROOT/venvs/native-runtime/bin/python
INDEX=$ART/artifact-index.json
for model in dp_cnn dp_transformer ibc lstm_gmm; do
  for seed in 0 1 2; do
    bundle_dir="$ART/models/200ep/$model/seed-$seed/bundle"
    if ! test -f "$bundle_dir/bundle_manifest.json"; then
      echo "SKIP_NO_BUNDLE $model seed-$seed"
      continue
    fi
    artifact=$(printf "%s-200ep-seed-%s" "$model" "$seed" | tr _ -)
    if jq -e --arg a "$artifact" '.artifacts[$a].metrics_path != null' "$INDEX" >/dev/null 2>&1; then
      echo "SKIP_EXISTS $artifact"
      continue
    fi
    output="$ART/evaluations/200ep/$model/seed-$seed"
    rm -rf "$output"
    cd "$PKG"
    export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1
    export PYTHONPYCACHEPREFIX="$DF_ROOT/python-cache"
    export PYTHONPATH=$PKG/src:$ART/cache/upstream/stanford:$ART/cache/upstream/robomimic
    export WANDB_MODE=offline
    "$NATIVE" -m so101_pusht_benchmark.cli evaluate-model \
      --model "$model" \
      --bundle "$bundle_dir/policy.safetensors" \
      --output "$output" \
      --artifact-id "$artifact" \
      --artifact-index "$INDEX"
    echo "EVALUATED $artifact"
  done
done
