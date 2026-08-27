"""Run one authentic two-sample physical-frame policy inference without actuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from so101_pusht_benchmark.hardware_profile import load_hardware_profile
import sys
from typing import cast

from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    ReceiptRoutingError,
    prepare_receipt_directory,
    validate_receipt_path,
)
from so101_pusht_benchmark.sim_to_real.replay_history import (
    HistoryEvidence,
    build_history,
    build_receipt,
    receipt_digest,
    validate_camera_receipt,
    validate_joint_receipt,
    validate_lineage_receipt,
)
from so101_pusht_benchmark.sim_to_real.replay_types import HistoryStep, InferenceReceipt
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

ARTIFACT_IDS = {"dp_cnn": "local-dp_cnn-recovered-v3-seed0"}
LINEAGE_AUTHORITY_DIGEST = "192d568795b756ac1edcde78a4a24ed8d37f1fef3bde14cd32a6d441c221a5e4"


class ShadowCLIError(RuntimeError):
    """Fail-closed CLI boundary error with a stable rollout code."""

    def __init__(self, code: RolloutCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=tuple(ARTIFACT_IDS))
    parser.add_argument("--artifact")
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument(
        "--lineage-authority-digest",
        default=LINEAGE_AUTHORITY_DIGEST,
        help="Expected signed lineage authority digest; fixture default retained for tests.",
    )
    parser.add_argument("--joint", required=True, type=Path)
    parser.add_argument("--camera", required=True, type=Path)
    parser.add_argument("--hardware-profile", type=Path)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _json_mapping(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShadowCLIError(RolloutCode.R_MISSING, f"cannot read {path}") from exc
    if not isinstance(raw, dict):
        raise ShadowCLIError(RolloutCode.R_MISSING, f"{path} must be a JSON mapping")
    return cast("dict[str, object]", raw)


def _samples(document: dict[str, object]) -> tuple[dict[str, object], ...]:
    raw = document.get("samples")
    if not isinstance(raw, list):
        raise ShadowCLIError(RolloutCode.HISTORY_INCOMPLETE, "samples list is missing")
    records: list[dict[str, object]] = []
    for item in cast("list[object]", raw):
        if not isinstance(item, dict):
            raise ShadowCLIError(RolloutCode.HISTORY_INCOMPLETE, "sample must be a mapping")
        records.append(cast("dict[str, object]", item))
    return tuple(records)


def _receipt_document(
    args: argparse.Namespace,
    history: tuple[HistoryStep, HistoryStep],
    inference: InferenceReceipt,
) -> dict[str, object]:
    inference_document = inference.to_document()
    latest = history[-1]
    return {
        "schema": 2,
        "mode": "physical_frame_shadow_only",
        "actuation_performed": False,
        "follower_motor_writes_performed": False,
        "follower_actuation_performed": False,
        "model": args.model,
        "artifact_id": inference_document["artifact_id"],
        "artifact_root": str(args.artifact_root.resolve()),
        "evidence_scope": (
            "test_fixture_only"
            if inference.policy == "fixture_deterministic_adapter"
            else "production"
        ),
        "policy_evidence": (
            "fixture_adapter_not_frozen_production"
            if inference.policy == "fixture_deterministic_adapter"
            else "authentic_frozen_production"
        ),
        "frame": str(args.frame.resolve()),
        "frame_sha256": latest.camera_sha256,
        "checkpoint_image_contract": "CCW90 RGB uint8[96,96,3]",
        "agent_pos": latest.agent_pos.tolist(),
        "agent_pos_source": "two_sample_receipt_bound_affine_mapping",
        "sample_ids": inference_document["sample_ids"],
        "sample_digests": inference_document["sample_digests"],
        "camera_sha256s": inference_document["camera_sha256s"],
        "agent_pos_sha256s": inference_document["agent_pos_sha256s"],
        "policy_seed": inference_document["seed"],
        "predicted_actions": inference_document["action_chunk_float32_2d"],
        "inference_digest": receipt_digest(inference),
        "action_semantics": "simulator absolute_mocap_xy; never sent to robot",
        "deployment_valid": False,
    }


def main() -> int:
    args = parse_args()
    try:
        if not args.artifact_root.is_dir():
            raise ShadowCLIError(RolloutCode.R_MISSING, "artifact root is unavailable")
        lineage_document = _json_mapping(args.lineage)
        lineage = validate_lineage_receipt(
            lineage_document,
            expected_digest=args.lineage_authority_digest,
        )
        artifact_id = args.artifact or ARTIFACT_IDS[args.model]
        if artifact_id != lineage["artifact_id"]:
            raise ShadowCLIError(RolloutCode.R_HASH_MISMATCH, "artifact and lineage differ")
        joint_document = _json_mapping(args.joint)
        camera_document = _json_mapping(args.camera)
        if args.hardware_profile is None:
            joint_digest = validate_joint_receipt(joint_document)
            camera_digest = validate_camera_receipt(camera_document)
        else:
            profile = load_hardware_profile(args.hardware_profile)
            joint_digest = validate_joint_receipt(
                joint_document, expected_digest=profile.joint_equivalence_digest
            )
            camera_digest = validate_camera_receipt(
                camera_document,
                expected_digest=profile.camera_registration_digest,
                expected_scope="authorized_physical_diagnostic",
            )
        history = build_history(
            HistoryEvidence(
                samples=_samples(_json_mapping(args.samples)),
                joint_document=joint_document,
                camera_document=camera_document,
                lineage_document=lineage_document,
                lineage_authority_digest=args.lineage_authority_digest,
                source_frame_path=args.frame.resolve(),
            )
        )
        inference = build_receipt(
            history,
            lineage=lineage,
            joint_digest=joint_digest,
            camera_digest=camera_digest,
            policy_seed=args.policy_seed,
        )
        document = _receipt_document(args, history, inference)
        production = inference.policy == "frozen"
        output = validate_receipt_path(args.output, production=production)
        prepare_receipt_directory(output.parent, production=production)
    except (ReceiptRoutingError, RolloutViolation, ShadowCLIError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
