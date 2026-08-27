"""Fixed evidence inventory for guided manual joint/FK corpus capture."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Final

from .joint_mapping import JOINT_ORDER

JOINT_CORPUS_PROVIDER_DIGEST: Final = hashlib.sha256(
    b"so101-pusht-benchmark:guided-direct-bus-joint-corpus-provider:v1"
).hexdigest()
TORQUE_PERMISSION: Final = "sync_read:Torque_Enable"
TORQUE_NOT_PROVEN_INSTRUCTION: Final = (
    "STOP: torque state is not proven under the signed authority. Do not force or manually "
    "move any joint. Ask the owner to issue a current read-only authority containing follower "
    "permission 'sync_read:Torque_Enable', then rerun --preflight-only."
)
TORQUE_RESISTING_INSTRUCTION: Final = (
    "STOP: torque is enabled or a resisting state is possible. Do not force or manually move "
    "any joint. Use a separately authorized torque-disable procedure; this CLI will not change "
    "torque. Then rerun --preflight-only."
)
MIN_ISOLATED_DELTA_DEGREES: Final = 5.0
MAX_OTHER_JOINT_DRIFT_DEGREES: Final = 3.0
MIN_DISTINCT_POSE_DELTA_DEGREES: Final = 1.0


@dataclass(frozen=True, slots=True)
class PoseInstruction:
    """One ordered, operator-confirmed corpus member."""

    identifier: str
    split: str
    category: str
    instruction: str
    isolated_joint: str | None = None
    direction: int = 0


POSE_PLAN: Final = (
    PoseInstruction(
        "fit-baseline",
        "fit",
        "baseline",
        "Place the arm in a comfortable neutral pose above the task plane, away from every hard stop.",
    ),
    *tuple(
        PoseInstruction(
            f"fit-{joint}-neg",
            "fit",
            "isolated",
            f"From baseline, move only {joint} in its negative direction by at least 5 degrees.",
            joint,
            -1,
        )
        for joint in JOINT_ORDER
    ),
    *tuple(
        PoseInstruction(
            f"fit-{joint}-pos",
            "fit",
            "isolated",
            f"From baseline, move only {joint} in its positive direction by at least 5 degrees.",
            joint,
            1,
        )
        for joint in JOINT_ORDER
    ),
    PoseInstruction(
        "fit-task-left",
        "fit",
        "task_plane",
        "Place the tool over the left task-plane region using a comfortable combination of at least two joints.",
    ),
    PoseInstruction(
        "fit-task-right",
        "fit",
        "task_plane",
        "Place the tool over the right task-plane region using a different comfortable combination of at least two joints.",
    ),
    PoseInstruction(
        "held-task-a",
        "held_out",
        "task_plane",
        "Choose a new central task-plane combination not used by either fit task pose.",
    ),
    PoseInstruction(
        "held-task-b",
        "held_out",
        "task_plane",
        "Choose a second new task-plane combination, distinct from every earlier pose.",
    ),
)


def pose_plan_digest() -> str:
    """Bind resumable sessions to the exact ordered pose inventory."""
    payload = [
        {
            "id": pose.identifier,
            "split": pose.split,
            "category": pose.category,
            "isolated_joint": pose.isolated_joint,
            "direction": pose.direction,
        }
        for pose in POSE_PLAN
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def render_operator_checklist() -> str:
    """Generate the field checklist from the machine-consumed pose plan."""
    lines = [
        "# Guided SO-101 joint/FK corpus checklist",
        "",
        "- This workflow never moves the robot and never writes motor, torque, config, or calibration registers.",
        "- Run `--preflight-only` first. Continue only when every motor reports torque disabled.",
        "- Never force a resisting joint. Stop and use a separately authorized torque-disable procedure.",
        "- Position one pose at a time; type the exact displayed `CAPTURE <pose-id>` confirmation only after hands are clear.",
        "- `STOP` preserves a resumable partial session. A partial session is never complete evidence.",
        "",
        "## Current evidence/authority blocker",
        "",
        "The signed `final-204-member-798e9829-20260825T043617Z.json` receipt contains two genuine but static Present_Position samples. It proves the 204-member source lineage and read-only acquisition path, not multi-pose equivalence. Its v5 authority lacks `sync_read:Torque_Enable` and carries the earlier synchronized-sample provider identity, so it cannot authorize manual positioning.",
        "",
        f"The fail-closed instruction is: `{TORQUE_NOT_PROVEN_INSTRUCTION}`",
        "",
        "## Ordered poses",
        "",
    ]
    for index, pose in enumerate(POSE_PLAN, 1):
        lines.append(f"{index}. `{pose.identifier}` - {pose.instruction}")
    lines.extend(
        [
            "",
            "## Publication boundary",
            "",
            "Completion creates an unsigned exact-member corpus candidate only. It becomes genuine governed evidence only after the owner signs the generated corpus binding and the governed physical auditor publishes a canonical receipt.",
            "",
            "## Exact production command sequence",
            "",
            "```bash",
            "cd /home/intro/InternLab/02_InTro_Project/03_code/so101_pusht_benchmark",
            "export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl",
            "export PY=/home/intro/miniforge3/envs/so100test/bin/python",
            "export ROOT=/home/intro/InternLab/02_InTro_Project/04_experiments/so101_pusht_benchmark/inference/sim_to_real_rollout",
            "export CAPTURE_ID=joint-fk-$(date -u +%Y%m%dT%H%M%SZ)",
            "export SESSION=$ROOT/joint-equivalence/$CAPTURE_ID",
            "export AUTH_DIR=$ROOT/authority/inputs/joint-corpus-owner-approved-20260825T051840Z",
            "export POS_DIR=$AUTH_DIR",
            "export POLICY=$AUTH_DIR/joint-corpus-production-policy.yaml",
            "",
            "$PY scripts/capture_joint_fk_corpus.py --live --profile configs/hardware/so101_real_v1.yaml --acquisition-authority $AUTH_DIR/read-only-acquisition-authority.json --authority-signature $AUTH_DIR/read-only-acquisition-authority.sig --positioning-authority $POS_DIR/manual-positioning-authority.json --positioning-signature $POS_DIR/manual-positioning-authority.sig --trust-anchor $AUTH_DIR/owner-trust-anchor.pem --preflight-only",
            "",
            "$PY scripts/capture_joint_fk_corpus.py --live --profile configs/hardware/so101_real_v1.yaml --policy $POLICY --acquisition-authority $AUTH_DIR/read-only-acquisition-authority.json --authority-signature $AUTH_DIR/read-only-acquisition-authority.sig --positioning-authority $POS_DIR/manual-positioning-authority.json --positioning-signature $POS_DIR/manual-positioning-authority.sig --trust-anchor $AUTH_DIR/owner-trust-anchor.pem --session-dir $SESSION --capture-id $CAPTURE_ID",
            "",
            "$PY scripts/prepare_joint_corpus_authority.py prepare --corpus $SESSION --trust-anchor $AUTH_DIR/owner-trust-anchor.pem --approval-id $CAPTURE_ID-owner-approval --output-dir $SESSION",
            'openssl dgst -sha256 -sign "$OWNER_PRIVATE_KEY" -out $SESSION/corpus-authority-binding.sig $SESSION/corpus-authority-binding.json',
            "$PY scripts/prepare_joint_corpus_authority.py assemble --request $SESSION/corpus-authority-request.json --binding $SESSION/corpus-authority-binding.json --signature $SESSION/corpus-authority-binding.sig --trust-anchor $AUTH_DIR/owner-trust-anchor.pem --output $SESSION/corpus-authority.json",
            "$PY scripts/audit_joint_equivalence_read_only.py --governed-physical --corpus $SESSION --policy $POLICY --corpus-authority $SESSION/corpus-authority.json --trust-anchor $AUTH_DIR/owner-trust-anchor.pem --output $SESSION/joint-equivalence.json",
            "```",
            "",
            f"The fresh supplemental positioning authority must contain provider digest `{JOINT_CORPUS_PROVIDER_DIGEST}` and permissions `direct_bus_connect`, `sync_read:Torque_Enable`, `sync_read:Present_Position`, and `disconnect:disable_torque=false`. It is signed by the same persistent owner key and binds the current base acquisition authority, lineage, follower, and calibration digests. The owner private key is never passed to a Python CLI or stored in the corpus.",
            "",
        ]
    )
    return "\n".join(lines)
