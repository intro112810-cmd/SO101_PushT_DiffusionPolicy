"""Boundary parsing for provenance receipt documents.

Owns the validation seam for physical samples plus the audited joint, camera,
and lineage receipts. Untrusted JSON is turned into typed values here; interior
code receives only validated structures.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import cast

from so101_pusht_benchmark.sim_to_real.joint_mapping import JOINT_ORDER
from so101_pusht_benchmark.sim_to_real.replay_types import (
    CAMERA_REGISTRATION_DIGEST,
    EXECUTED_ACTIONS,
    FIXTURE_LINEAGE_ID,
    JOINT_EQUIVALENCE_DIGEST,
    PRODUCTION_LINEAGE_ID,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

HistoryDocument = Mapping[str, object]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, f"{label} must be a mapping")
    return cast("dict[str, object]", value)


def require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"{label} must be SHA-256")
    result = value.lower()
    if any(character not in "0123456789abcdef" for character in result):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"{label} must be SHA-256")
    return result


def require_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RolloutViolation(RolloutCode.R_NONFINITE, f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RolloutViolation(RolloutCode.R_NONFINITE, f"{label} must be finite")
    return result


def require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RolloutViolation(RolloutCode.R_NONFINITE, f"{label} must be an integer")
    return value


def require_action_rows(value: object, label: str) -> list[list[float]]:
    rows = cast("list[object]", value) if isinstance(value, list) else []
    if len(rows) != EXECUTED_ACTIONS:
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, f"{label} count")
    parsed: list[list[float]] = []
    for row in rows:
        typed_row = cast("list[object]", row) if isinstance(row, list) else []
        if len(typed_row) != 2:
            raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, f"{label} width")
        parsed.append([require_float(item, label) for item in typed_row])
    return parsed


def canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def parse_sample_document(raw: Mapping[str, object]) -> dict[str, object]:
    """Parse one accepted physical-sample record from the capture receipt."""
    kind = raw.get("kind")
    if kind != "physical_sample":
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "sample kind is invalid")
    record_id = raw.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "sample id is missing")
    created_at = require_float(raw.get("created_at"), "sample created_at")
    camera_timestamp = require_float(raw.get("camera_timestamp"), "camera_timestamp")
    joint_timestamp = require_float(raw.get("joint_timestamp"), "joint_timestamp")
    digest = require_digest(raw.get("digest"), "sample digest")
    frame_digest = require_digest(raw.get("frame_digest"), "frame_digest")
    body_degrees = (
        cast("list[object]", raw.get("body_degrees"))
        if isinstance(raw.get("body_degrees"), list)
        else []
    )
    if len(body_degrees) != 5:
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "body_degrees must be five values")
    body = tuple(require_float(item, "body_degrees") for item in body_degrees)
    device_digest = require_digest(raw.get("device_digest"), "device_digest")
    calibration_digest = require_digest(raw.get("calibration_digest"), "calibration_digest")
    record = {
        "kind": "physical_sample",
        "record_id": record_id,
        "created_at": created_at,
        "camera_timestamp": camera_timestamp,
        "joint_timestamp": joint_timestamp,
        "frame_digest": frame_digest,
        "body_degrees": list(body),
        "device_digest": device_digest,
        "calibration_digest": calibration_digest,
    }
    if sha256_bytes(canonical(record)) != digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "sample content")
    return {**record, "digest": digest}


def validate_camera_receipt(
    receipt: Mapping[str, object],
    *,
    expected_digest: str | None = CAMERA_REGISTRATION_DIGEST,
    expected_scope: str = "synthetic_test_fixture",
) -> str:
    """Require an explicitly bound geometric camera registration digest."""
    if receipt.get("audited") is not True:
        raise RolloutViolation(RolloutCode.R_CAMERA_EQUIVALENCE_UNPROVEN, "camera not audited")
    declared = require_digest(receipt.get("digest"), "camera digest")
    if expected_digest is not None and declared != require_digest(
        expected_digest, "expected camera digest"
    ):
        raise RolloutViolation(RolloutCode.R_CAMERA_EQUIVALENCE_UNPROVEN, "camera hash drift")
    if (
        receipt.get("evidence_scope") != expected_scope
        or receipt.get("metrics_source") != "recomputed_from_raw_points_and_matrices"
    ):
        raise RolloutViolation(
            RolloutCode.R_CAMERA_EQUIVALENCE_UNPROVEN,
            "camera receipt lacks recomputed raw-evidence provenance",
        )
    for field in (
        "fit_correspondences",
        "held_out_correspondences",
        "fit_reprojection_error_px",
        "held_out_reprojection_error_px",
        "max_correspondence_error_px",
        "fit_physical_to_sim_residual_m",
        "held_out_physical_to_sim_residual_m",
        "max_physical_to_sim_residual_m",
        "member_digests",
        "checkpoint_view_members",
        "device_hash",
        "config_hash",
        "orientation_hash",
    ):
        if field not in receipt:
            raise RolloutViolation(RolloutCode.R_CAMERA_EQUIVALENCE_UNPROVEN, f"missing {field}")
    if any(
        require_int(receipt.get(field), field) < 12
        for field in ("fit_correspondences", "held_out_correspondences")
    ):
        raise RolloutViolation(
            RolloutCode.R_CAMERA_EQUIVALENCE_UNPROVEN, "camera correspondences incomplete"
        )
    member_digests = receipt.get("member_digests")
    if not isinstance(member_digests, Mapping):
        raise RolloutViolation(
            RolloutCode.R_CAMERA_EQUIVALENCE_UNPROVEN, "camera raw-member identities incomplete"
        )
    typed_member_digests = cast("Mapping[object, object]", member_digests)
    if len(typed_member_digests) < 4:
        raise RolloutViolation(
            RolloutCode.R_CAMERA_EQUIVALENCE_UNPROVEN, "camera raw-member identities incomplete"
        )
    for member_id, digest in typed_member_digests.items():
        if not isinstance(member_id, str) or not member_id:
            raise RolloutViolation(
                RolloutCode.R_CAMERA_EQUIVALENCE_UNPROVEN, "camera member identity invalid"
            )
        require_digest(digest, f"camera member {member_id}")
    checkpoint_members = receipt.get("checkpoint_view_members")
    if (
        not isinstance(checkpoint_members, list)
        or len(cast("list[object]", checkpoint_members)) < 2
    ):
        raise RolloutViolation(
            RolloutCode.R_CAMERA_EQUIVALENCE_UNPROVEN, "checkpoint-view coverage incomplete"
        )
    for field in (
        "fit_reprojection_error_px",
        "held_out_reprojection_error_px",
        "max_correspondence_error_px",
        "fit_physical_to_sim_residual_m",
        "held_out_physical_to_sim_residual_m",
        "max_physical_to_sim_residual_m",
    ):
        if require_float(receipt.get(field), field) < 0:
            raise RolloutViolation(
                RolloutCode.R_CAMERA_EQUIVALENCE_UNPROVEN, "camera residual cannot be negative"
            )
    return declared


def validate_joint_receipt(
    receipt: Mapping[str, object],
    *,
    expected_digest: str | None = JOINT_EQUIVALENCE_DIGEST,
) -> str:
    """Require an explicitly bound multi-pose joint-frame equivalence digest."""
    if receipt.get("audited") is not True:
        raise RolloutViolation(RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, "joints not audited")
    order = receipt.get("joint_order")
    if not isinstance(order, list):
        raise RolloutViolation(RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, "joint order mismatch")
    typed_order = cast("list[object]", order)
    if tuple(typed_order) != JOINT_ORDER:
        raise RolloutViolation(RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, "joint order mismatch")
    declared = require_digest(receipt.get("digest"), "joint digest")
    if expected_digest is not None and declared != require_digest(
        expected_digest, "expected joint digest"
    ):
        raise RolloutViolation(RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, "joint hash drift")
    for field in (
        "fit_count",
        "held_out_count",
        "task_plane_pose_count",
        "max_fk_residual_m",
    ):
        if field not in receipt:
            raise RolloutViolation(RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, f"missing {field}")
    if (
        require_int(receipt.get("fit_count"), "fit_count") < 2
        or require_int(receipt.get("held_out_count"), "held_out_count") < 1
    ):
        raise RolloutViolation(RolloutCode.R_JOINT_EQUIVALENCE_UNPROVEN, "single-pose evidence")
    return declared


def validate_lineage_receipt(
    receipt: Mapping[str, object],
    *,
    expected_digest: str,
) -> dict[str, object]:
    """Validate one compact fixture lineage document without touching the 400k bundle."""
    document = {
        "artifact_id": receipt.get("artifact_id"),
        "authority_digest": receipt.get("authority_digest"),
        "valid": receipt.get("valid"),
    }
    artifact_id = document["artifact_id"]
    if artifact_id not in {FIXTURE_LINEAGE_ID, PRODUCTION_LINEAGE_ID}:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "lineage artifact identity")
    if document["valid"] is not True:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "lineage is not valid")
    lineage_digest = sha256_bytes(canonical(document))
    declared = require_digest(receipt.get("lineage_digest"), "lineage_digest")
    if declared != lineage_digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "lineage receipt digest")
    authority = require_digest(receipt.get("authority_digest"), "authority_digest")
    if authority != require_digest(expected_digest, "lineage authority"):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "lineage authority identity drift")
    return {
        "artifact_id": artifact_id,
        "authority_digest": authority,
        "lineage_digest": declared,
        "fixture_only": artifact_id == FIXTURE_LINEAGE_ID,
    }
