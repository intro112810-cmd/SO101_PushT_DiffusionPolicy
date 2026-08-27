#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  server_systemd_training.sh start MODEL SEED PAPER_VIEW OUTPUT ARTIFACT_ID ARTIFACT_INDEX PAPER_PROFILES MAX_UPDATES
  server_systemd_training.sh dry-run MODEL SEED PAPER_VIEW OUTPUT ARTIFACT_ID ARTIFACT_INDEX PAPER_PROFILES MAX_UPDATES
  server_systemd_training.sh status MODEL
  server_systemd_training.sh logs MODEL
  server_systemd_training.sh stop MODEL

MODEL: dp_cnn | dp_transformer | ibc | lstm_gmm
All paths must be absolute and beneath /home/user/kihyun/df.
EOF
  exit 2
}

[[ $# -ge 2 ]] || usage
ACTION=$1
MODEL=$2

case "$MODEL" in
  dp_cnn | dp_transformer | ibc | lstm_gmm) ;;
  *) usage ;;
esac

SEED=${3:-}
if [[ "$ACTION" == "start" || "$ACTION" == "dry-run" ]]; then
  [[ "$SEED" =~ ^[012]$ ]] || usage
fi
UNIT="kihyun-pusht-${MODEL//_/-}${SEED:+-seed-$SEED}-train"
SAMPLER_UNIT="${UNIT}-resources"
DF_ROOT=/home/user/kihyun/df
PYTHON_CACHE_ROOT="$DF_ROOT/python-cache"
RESOURCE_LOG="$DF_ROOT/resource-logs/$UNIT.jsonl"
PROJECT_ROOT=$DF_ROOT/02_InTro_Project
PACKAGE_ROOT=$PROJECT_ROOT/03_code/so101_pusht_benchmark
PYTHON=$DF_ROOT/venvs/paper-baselines/bin/python

require_owned_path() {
  local path=$1
  [[ "$path" = /* ]] || {
    echo "FAIL CLOSED: path must be absolute: $path" >&2
    exit 2
  }
  [[ "$path" == "$DF_ROOT"/* ]] || {
    echo "FAIL CLOSED: path is outside $DF_ROOT: $path" >&2
    exit 2
  }
  [[ "$path" != *"/../"* && "$path" != */.. ]] || {
    echo "FAIL CLOSED: path traversal is forbidden: $path" >&2
    exit 2
  }
}

case "$ACTION" in
  status)
    [[ $# -eq 2 ]] || usage
    systemctl --user show "$UNIT.service" \
      --property=LoadState,ActiveState,SubState,Result,ExecMainPID,ExecMainStatus
    if [[ -f "$DF_ROOT/service-receipts/$UNIT.json" ]]; then
      cat "$DF_ROOT/service-receipts/$UNIT.json"
    fi
    ;;
  logs)
    [[ $# -eq 2 ]] || usage
    journalctl --user-unit "$UNIT.service" --no-pager -n 100
    ;;
  stop)
    [[ $# -eq 2 ]] || usage
    if systemctl --user is-active --quiet "$UNIT.service"; then
      systemctl --user kill --signal=SIGINT "$UNIT.service"
      systemctl --user stop "$SAMPLER_UNIT.service" 2>/dev/null || true
      echo "SIGINT_SENT: $UNIT.service"
    else
      echo "NOT_ACTIVE: $UNIT.service"
    fi
    mkdir -p "$DF_ROOT/resource-logs" "$DF_ROOT/service-receipts"
    ;;
  start | dry-run)
    [[ $# -eq 9 ]] || usage
    PAPER_VIEW=$4
    OUTPUT=$5
    ARTIFACT_ID=$6
    ARTIFACT_INDEX=$7
    PAPER_PROFILES=$8
    MAX_UPDATES=$9
    [[ "$MAX_UPDATES" =~ ^[1-9][0-9]*$ ]] || usage
    require_owned_path "$PAPER_VIEW"
    require_owned_path "$OUTPUT"
    require_owned_path "$ARTIFACT_INDEX"
    require_owned_path "$PAPER_PROFILES"
    require_owned_path "$PACKAGE_ROOT"
    require_owned_path "$PYTHON"
    [[ -d "$PAPER_VIEW" ]] || {
      echo "FAIL CLOSED: paper view does not exist: $PAPER_VIEW" >&2
      exit 1
    }
    [[ ! -e "$OUTPUT" ]] || {
      echo "FAIL CLOSED: output already exists: $OUTPUT" >&2
      exit 1
    }
    [[ -f "$ARTIFACT_INDEX" ]] || {
      echo "FAIL CLOSED: artifact index does not exist: $ARTIFACT_INDEX" >&2
      exit 1
    }
    [[ -x "$PYTHON" ]] || {
      echo "FAIL CLOSED: isolated paper runtime missing: $PYTHON" >&2
      exit 1
    }
    if systemctl --user is-active --quiet "$UNIT.service"; then
      echo "FAIL CLOSED: service already active: $UNIT.service" >&2
      exit 1
    fi

    command=(
      "$PYTHON" -m so101_pusht_benchmark.cli train-model
      --model "$MODEL"
      --paper-view "$PAPER_VIEW"
      --output "$OUTPUT"
      --artifact-id "$ARTIFACT_ID"
      --artifact-index "$ARTIFACT_INDEX"
      --paper-profiles "$PAPER_PROFILES"
      --seed "$SEED"
      --full-production
      --max-updates "$MAX_UPDATES"
    )
    systemd_command=(
      systemd-run --user
      --unit "$UNIT"
      --description "Kihyun PushT ${MODEL} production training"
      --property "WorkingDirectory=$PACKAGE_ROOT"
      --property "Restart=no"
      --property "KillSignal=SIGINT"
      --property "TimeoutStopSec=300"
      --setenv "CUDA_VISIBLE_DEVICES=0"
      --setenv "MUJOCO_GL=egl"
      --setenv "PYTHONDONTWRITEBYTECODE=1"
      --setenv "PYTHONPYCACHEPREFIX=$PYTHON_CACHE_ROOT"
      --setenv "PYTHONPATH=$PACKAGE_ROOT/src"
      --setenv "WANDB_MODE=offline"
      "${command[@]}"
    )

    if [[ "$ACTION" == "dry-run" ]]; then
      printf 'DRY_RUN:'
      printf ' %q' "${systemd_command[@]}"
      printf '\n'
      echo "NOTE: user manager Linger must be checked before a long run."
    else
      systemd-run --user --unit "${UNIT}-preflight" \
        --property "WorkingDirectory=$PACKAGE_ROOT" /usr/bin/true
      systemctl --user is-failed --quiet "${UNIT}-preflight.service" && {
        echo "FAIL CLOSED: user-systemd preflight failed" >&2
        exit 1
      }
      "${systemd_command[@]}"
      systemctl --user is-active --quiet "$UNIT.service" || {
        echo "FAIL CLOSED: service did not become active: $UNIT.service" >&2
        exit 1
      }
      main_pid=$(systemctl --user show "$UNIT.service" --property=MainPID --value)
      systemd-run --user \
        --unit "$SAMPLER_UNIT" \
        --description "Kihyun PushT ${MODEL} resource sampler" \
        --property "WorkingDirectory=$PACKAGE_ROOT" \
        --property "Restart=no" \
        "$PYTHON" "$PACKAGE_ROOT/scripts/server_resource_sampler.py" \
        --pid "$main_pid" \
        --output "$RESOURCE_LOG" \
        --interval 30
      systemd-run --user \
        --unit "${UNIT}-receipt-writer" \
        --description "Kihyun PushT ${MODEL} service receipt writer" \
        --property "WorkingDirectory=$PACKAGE_ROOT" \
        --property "Restart=no" \
        /bin/bash -lc \
        "while systemctl --user is-active --quiet '$UNIT.service'; do read -r -t 30 _ || true; done; exec '$PYTHON' '$PACKAGE_ROOT/scripts/write_service_receipt.py' --unit '$UNIT.service' --output '$DF_ROOT/service-receipts/$UNIT.json' --model-output '$OUTPUT' --resource-log '$RESOURCE_LOG'"
      echo "ACTIVE: $UNIT.service"
      systemctl --user show "$UNIT.service" \
        --property=ActiveState,SubState,ExecMainPID
    fi
    ;;
  *) usage ;;
esac
