#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/intro/InternLab/02_InTro_Project
CODE="$PROJECT/03_code/so101_pusht_benchmark"
ROOT="$PROJECT/04_experiments/so101_pusht_benchmark/inference/sim_to_real_rollout"
AUTH="$ROOT/authority/inputs/lab-joint-readiness-20260827"
PYTHON=/home/intro/miniforge3/envs/so100test/bin/python

cd "$CODE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src

exec "$PYTHON" scripts/capture_joint_fk_corpus.py \
  --live \
  --profile configs/hardware/so101_real_v1.yaml \
  --policy "$AUTH/joint-corpus-production-policy.yaml" \
  --acquisition-authority "$AUTH/read-only-acquisition-authority.json" \
  --authority-signature "$AUTH/read-only-acquisition-authority.sig" \
  --positioning-authority "$AUTH/manual-positioning-authority.json" \
  --positioning-signature "$AUTH/manual-positioning-authority.sig" \
  --trust-anchor "$AUTH/owner-trust-anchor.pem" \
  --session-dir "$ROOT/joint/raw/joint-fk-corpus-lab-20260827" \
  --capture-id joint-fk-lab-20260827
