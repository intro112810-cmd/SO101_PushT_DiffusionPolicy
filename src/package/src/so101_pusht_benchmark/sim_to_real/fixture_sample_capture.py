"""Pinned fixture-stream loading for non-genuine sample-capture tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from .policy_parser import load_fixture_safety_policy
from .rollout_codes import RolloutCode, RolloutViolation
from .sample_capture import (
    JointState,
    StaticCameraSource,
    StaticJointSource,
    acquire_samples,
    load_policy_limits,
    sample_as_record,
)

__all__ = ("capture_fixture_receipt",)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, f"{label} must be a mapping")
    return cast("dict[str, object]", value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RolloutViolation(RolloutCode.R_NONFINITE, f"{label} must be a number")
    return float(value)


def _frame(value: object) -> bytes:
    if not isinstance(value, str):
        raise RolloutViolation(RolloutCode.R_NONFINITE, "frame is not hex text")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise RolloutViolation(RolloutCode.R_NONFINITE, "frame is not valid hex") from exc


def _joint(snapshot: dict[str, object]) -> JointState:
    raw = snapshot.get("body_degrees")
    if not isinstance(raw, list):
        raise RolloutViolation(RolloutCode.R_MISSING, "body_degrees must be five values")
    body_values = cast("list[object]", raw)
    if len(body_values) != 5:
        raise RolloutViolation(RolloutCode.R_MISSING, "body_degrees must be five values")
    values = tuple(_number(value, "body_degrees") for value in body_values)
    return JointState(
        _number(snapshot.get("joint_timestamp"), "joint_timestamp"),
        (values[0], values[1], values[2], values[3], values[4]),
        str(snapshot.get("device_digest", "")),
        str(snapshot.get("calibration_digest", "")),
    )


def _sources(
    directory: Path,
    count: int,
) -> tuple[StaticCameraSource, StaticJointSource]:
    path = directory / "manifest.json"
    try:
        root = _mapping(json.loads(path.read_text(encoding="utf-8")), "fixture manifest")
    except (OSError, json.JSONDecodeError) as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, f"cannot read {path}") from exc
    raw_snapshots = root.get("snapshots")
    if not isinstance(raw_snapshots, list):
        raise RolloutViolation(RolloutCode.R_MISSING, "fixture snapshot count mismatch")
    snapshot_values = cast("list[object]", raw_snapshots)
    if len(snapshot_values) != count:
        raise RolloutViolation(RolloutCode.R_MISSING, "fixture snapshot count mismatch")
    snapshots = [_mapping(value, "snapshot") for value in snapshot_values]
    frames = [_frame(snapshot.get("frame")) for snapshot in snapshots]
    timestamps = [
        _number(snapshot.get("camera_timestamp"), "camera_timestamp") for snapshot in snapshots
    ]
    states = [_joint(snapshot) for snapshot in snapshots]
    return StaticCameraSource(frames, timestamps), StaticJointSource(states, count)


def capture_fixture_receipt(
    fixture: Path,
    policy_path: Path,
    count: int,
) -> dict[str, object]:
    """Build a receipt that can never claim physical or production authority."""
    policy = load_fixture_safety_policy(policy_path)
    limits = load_policy_limits(
        policy.timing.sample_max_age_seconds,
        policy.timing.sample_max_skew_seconds,
    )
    camera, joint = _sources(fixture, count)
    samples = acquire_samples(camera, joint, count=count, policy=limits)
    return {
        "schema": 1,
        "mode": "sim_to_real_physical_sample_capture",
        "evidence_scope": "test_fixture_only",
        "genuine_physical_samples": False,
        "policy_evidence": "fixture_stream_not_genuine_physical_evidence",
        "count": len(samples),
        "policy_digest": policy.canonical_digest,
        "samples": [sample_as_record(sample) for sample in samples],
        "motor_writes_performed": False,
        "actuation_performed": False,
    }
