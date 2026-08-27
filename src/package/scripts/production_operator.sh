#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 {--dry-run|preflight|post-collection} /absolute/user-experiment.yaml" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
MODE=$1
EXPERIMENT_CONFIG=$2
[[ "$MODE" == "--dry-run" || "$MODE" == "preflight" || "$MODE" == "post-collection" ]] || usage
[[ "$EXPERIMENT_CONFIG" = /* ]] || { echo "experiment config must be absolute" >&2; exit 2; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
PACKAGE_ROOT=$PROJECT_ROOT/03_code/so101_pusht_benchmark
ARTIFACT_ROOT=$PROJECT_ROOT/04_experiments/so101_pusht_benchmark
DATASET_ROOT=$ARTIFACT_ROOT/datasets/pusht_so100_native
NATIVE_STORE=$ARTIFACT_ROOT/datasets/native_store
FROZEN_STORE=$ARTIFACT_ROOT/datasets/frozen_four_model
ARTIFACT_INDEX=$ARTIFACT_ROOT/artifact-index.json
REPORT_OUTPUT=$ARTIFACT_ROOT/reports/four-model-final
TASK_TEMP=$ARTIFACT_ROOT/tmp/production-operator
PAPER_PYTHON=$ARTIFACT_ROOT/cache/envs/paper-baselines/bin/python
STANFORD_ROOT=$ARTIFACT_ROOT/cache/upstream/stanford
ROBOMIMIC_ROOT=$ARTIFACT_ROOT/cache/upstream/robomimic
PACKAGE_PYTHONPATH=$PACKAGE_ROOT/src
NATIVE_PYTHONPATH=$PACKAGE_ROOT/src:$STANFORD_ROOT:$ROBOMIMIC_ROOT

MODELS=(dp_cnn dp_transformer ibc lstm_gmm)
FINAL_IDS=(dp-cnn-production dp-transformer-production ibc-production lstm-gmm-production)
SMOKE_IDS=(dp-cnn-production-smoke dp-transformer-production-smoke ibc-production-smoke lstm-gmm-production-smoke)

run_command() {
  if [[ "$MODE" == "--dry-run" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

index_value() {
  local artifact_id=$1 field=$2
  "$PAPER_PYTHON" - "$ARTIFACT_INDEX" "$artifact_id" "$field" <<'PY'
import json, sys
from pathlib import Path
index, artifact_id, field = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if not index.is_file():
    print("")
else:
    value = json.loads(index.read_text()).get("artifacts", {}).get(artifact_id, {}).get(field, "")
    print(value if isinstance(value, str) else "")
PY
}

native_cli() {
  run_command env PYTHONPATH="$PACKAGE_PYTHONPATH" conda run -n so100test \
    python -m so101_pusht_benchmark.cli "$@"
}

paper_cli() {
  run_command env PYTHONPATH="$PACKAGE_PYTHONPATH" WANDB_MODE=offline \
    "$PAPER_PYTHON" -m so101_pusht_benchmark.cli "$@"
}

if [[ "$MODE" == "preflight" || "$MODE" == "--dry-run" ]]; then
  cd "$PACKAGE_ROOT"
  native_cli inspect-env --native-pusht-so100
  native_cli collect-native --preflight --dataset-root "$DATASET_ROOT"
  native_cli export-native --preflight
  if [[ "$MODE" == "--dry-run" ]]; then
    run_command env PYTHONPATH="$PACKAGE_PYTHONPATH" conda run -n so100test \
      python -m so101_pusht_benchmark.cli collect-native --launch --dataset-root "$DATASET_ROOT"
  else
    printf 'OPERATOR_NEXT: cd %q && PYTHONPATH=%q conda run -n so100test python -m so101_pusht_benchmark.cli collect-native --launch --dataset-root %q\n' \
      "$PACKAGE_ROOT" "$PACKAGE_PYTHONPATH" "$DATASET_ROOT"
    exit 0
  fi
fi

# --dry-run prints the complete post-collection route; post-collection executes it.
if [[ "$MODE" == "post-collection" || "$MODE" == "--dry-run" ]]; then
  cd "$PACKAGE_ROOT"
  native_cli freeze-experiment --metadata "$DATASET_ROOT/meta/info.json" \
    --experiment-config "$EXPERIMENT_CONFIG" --dry-run

  if [[ "$MODE" == "--dry-run" || ! -e "$NATIVE_STORE" ]]; then
    native_cli import-native --repo "$DATASET_ROOT" --output "$NATIVE_STORE"
  else
    echo "FAIL CLOSED: existing imported store has no authenticated resume receipt and was preserved: $NATIVE_STORE" >&2
    echo "RECOVERY: inspect it, move it to a new quarantine path, then rerun post-collection." >&2
    exit 1
  fi
  native_cli freeze-experiment --source "$NATIVE_STORE" --output "$FROZEN_STORE" \
    --experiment-config "$EXPERIMENT_CONFIG"

  if [[ "$MODE" != "--dry-run" ]]; then
    mkdir -p "$ARTIFACT_ROOT" "$ARTIFACT_ROOT/reports" "$TASK_TEMP"
    if [[ ! -e "$ARTIFACT_INDEX" ]]; then
      printf '{"schema":1,"artifacts":{}}\n' >"$ARTIFACT_INDEX"
    fi
  fi

  # Gate 1: all four one-update production smokes are non-final proof artifacts.
  for index in 0 1 2 3; do
    model=${MODELS[$index]}
    smoke_id=${SMOKE_IDS[$index]}
    smoke_output=$ARTIFACT_ROOT/smoke/$model
    status=""
    [[ "$MODE" == "--dry-run" ]] || status=$(index_value "$smoke_id" result_status)
    if [[ "$status" == "production_smoke_complete_nonfinal" ]]; then
      echo "FAIL CLOSED: non-final smoke cannot be skipped without authenticated resume validation: $smoke_id" >&2
      echo "RECOVERY: preserve and inspect $smoke_output; quarantine the incomplete artifact root before a clean rerun." >&2
      exit 1
    fi
    [[ -z "$status" ]] || { echo "unexpected smoke status for $smoke_id: $status" >&2; exit 1; }
    paper_cli train-model --model "$model" --paper-view "$FROZEN_STORE" \
      --output "$smoke_output" --artifact-id "$smoke_id" \
      --artifact-index "$ARTIFACT_INDEX" --smoke-mode production
  done

  # Gate 2: first incomplete full model. Final IDs remain identical through compare.
  for index in 0 1 2 3; do
    model=${MODELS[$index]}
    artifact_id=${FINAL_IDS[$index]}
    full_output=$ARTIFACT_ROOT/models/$model/full
    bundle_output=$ARTIFACT_ROOT/models/$model/bundle
    evaluation_output=$ARTIFACT_ROOT/evaluations/$model
    status=""
    [[ "$MODE" == "--dry-run" ]] || status=$(index_value "$artifact_id" result_status)

    if [[ -z "$status" ]]; then
      paper_cli train-model --model "$model" --paper-view "$FROZEN_STORE" \
        --output "$full_output" --artifact-id "$artifact_id" \
        --artifact-index "$ARTIFACT_INDEX" --full-production --max-updates 100000 --preflight
      paper_cli train-model --model "$model" --paper-view "$FROZEN_STORE" \
        --output "$full_output" --artifact-id "$artifact_id" \
        --artifact-index "$ARTIFACT_INDEX" --full-production --max-updates 100000
      status=full_training_complete
    fi

    if [[ "$status" == "full_training_complete" ]]; then
      native_cli validate-production-artifact --stage training --model "$model" \
        --artifact-id "$artifact_id" --artifact-index "$ARTIFACT_INDEX" --output "$full_output"
      [[ "$MODE" == "--dry-run" ]] || echo "RESUME: validated full training $artifact_id"
      checkpoint=$full_output/checkpoints/latest.ckpt
      config=$full_output/resolved_config.json
      paper_cli export-inference-bundle --checkpoint "$checkpoint" --config "$config" \
        --output "$bundle_output" --artifact-id "$artifact_id" \
        --artifact-index "$ARTIFACT_INDEX"
      status=full_training_bundle_ready
    fi

    if [[ "$status" == "full_training_bundle_ready" ]]; then
      native_cli validate-production-artifact --stage bundle --model "$model" \
        --artifact-id "$artifact_id" --artifact-index "$ARTIFACT_INDEX" --output "$bundle_output"
      [[ "$MODE" == "--dry-run" ]] || echo "RESUME: validated inference bundle $artifact_id"
      bundle=$bundle_output/policy.safetensors
      if [[ "$MODE" == "--dry-run" ]]; then
        run_command env MUJOCO_GL=egl PYTHONPATH="$NATIVE_PYTHONPATH" conda run -n so100test \
          python -m so101_pusht_benchmark.cli evaluate-model --model "$model" \
          --bundle "$bundle" --output "$evaluation_output" --artifact-id "$artifact_id" \
          --artifact-index "$ARTIFACT_INDEX" --device cuda:0
      else
        (
          cd "$TASK_TEMP"
          run_command env MUJOCO_GL=egl PYTHONPATH="$NATIVE_PYTHONPATH" conda run -n so100test \
            python -m so101_pusht_benchmark.cli evaluate-model --model "$model" \
            --bundle "$bundle" --output "$evaluation_output" --artifact-id "$artifact_id" \
            --artifact-index "$ARTIFACT_INDEX" --device cuda:0
        )
      fi
      status=anchored_final_evaluation
    fi

    [[ "$status" == "anchored_final_evaluation" ]] || {
      echo "unexpected final status for $artifact_id: $status" >&2
      exit 1
    }
    native_cli validate-production-artifact --stage evaluation --model "$model" \
      --artifact-id "$artifact_id" --artifact-index "$ARTIFACT_INDEX" \
      --output "$evaluation_output"
    if [[ "$MODE" == "--dry-run" ]]; then
      echo "DRY_RUN_NOTE: final evaluation completion will be claimed only after validation $artifact_id"
    else
      echo "RESUME: validated final evaluation $artifact_id"
    fi
  done

  if [[ "$MODE" == "--dry-run" || ! -e "$REPORT_OUTPUT" ]]; then
    native_cli compare-models --artifact-index "$ARTIFACT_INDEX" \
      --artifact-id dp-cnn-production --artifact-id dp-transformer-production \
      --artifact-id ibc-production --artifact-id lstm-gmm-production \
      --output "$REPORT_OUTPUT"
  else
    echo "RESUME: validating existing final report before reuse $REPORT_OUTPUT"
    if ! native_cli compare-models --artifact-index "$ARTIFACT_INDEX" \
      --artifact-id dp-cnn-production --artifact-id dp-transformer-production \
      --artifact-id ibc-production --artifact-id lstm-gmm-production \
      --output "$REPORT_OUTPUT" --validate-existing; then
      echo "FAIL CLOSED: existing final report was preserved and was not reused." >&2
      echo "RECOVERY: inspect it, move it to a new quarantine path outside $REPORT_OUTPUT, then rerun post-collection." >&2
      exit 1
    fi
    echo "RESUME: validated and reused byte-identical final report $REPORT_OUTPUT"
  fi
fi
