"""Boundary parsing for shadow-campaign sample fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from so101_pusht_benchmark.sim_to_real.replay_receipts import parse_sample_document
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.rollout_record_types import BodyDegrees, PhysicalSample
from so101_pusht_benchmark.sim_to_real.shadow_types import ShadowCampaignInput

__all__ = (
    "load_campaign_samples",
    "load_campaign_scene_pose",
    "sample_age_seconds",
    "samples_as_physical_samples",
)


def _json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, f"cannot read {label}") from exc
    if not isinstance(raw, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, f"{label} must be a JSON mapping")
    return cast("dict[str, object]", raw)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RolloutViolation(RolloutCode.R_NONFINITE, f"{label} must be numeric")
    return float(value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RolloutViolation(RolloutCode.R_MISSING, f"{label} must be a string")
    return value


def _body(value: object) -> BodyDegrees:
    if not isinstance(value, list):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "sample body degrees")
    typed = cast("list[object]", value)
    if len(typed) != 5:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "sample body degrees")
    numbers = [_number(item, "body degree") for item in typed]
    return numbers[0], numbers[1], numbers[2], numbers[3], numbers[4]


def _campaign_document(inputs: ShadowCampaignInput) -> dict[str, object]:
    return _json_mapping(inputs.fixture_dir / "samples.json", "campaign samples")


def load_campaign_scene_pose(inputs: ShadowCampaignInput) -> dict[str, object]:
    """Load the scene pose carried by the synchronized campaign fixture."""
    pose = _campaign_document(inputs).get("scene_pose")
    if not isinstance(pose, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, "campaign scene pose is missing")
    return cast("dict[str, object]", pose)


def load_campaign_samples(inputs: ShadowCampaignInput) -> tuple[dict[str, object], ...]:
    """Load and validate exactly two content-addressed physical samples."""
    raw_samples = _campaign_document(inputs).get("samples")
    if not isinstance(raw_samples, list):
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "campaign needs two samples")
    typed_samples = cast("list[object]", raw_samples)
    if len(typed_samples) != 2:
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "campaign needs two samples")
    result: list[dict[str, object]] = []
    for raw in typed_samples:
        if not isinstance(raw, dict):
            raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "campaign sample mapping")
        result.append(parse_sample_document(cast("dict[str, object]", raw)))
    return tuple(result)


def samples_as_physical_samples(
    records: tuple[dict[str, object], ...],
) -> tuple[PhysicalSample, PhysicalSample]:
    """Parse the validated sample records into supervisor-ready typed samples."""
    parsed = [
        PhysicalSample(
            _text(record["record_id"], "record_id"),
            _number(record["created_at"], "created_at"),
            _text(record["digest"], "digest"),
            _number(record["camera_timestamp"], "camera_timestamp"),
            _number(record["joint_timestamp"], "joint_timestamp"),
            _text(record["frame_digest"], "frame_digest"),
            _body(record["body_degrees"]),
            _text(record["device_digest"], "device_digest"),
            _text(record["calibration_digest"], "calibration_digest"),
        )
        for record in records
    ]
    if len(parsed) != 2:
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "campaign sample count")
    return parsed[0], parsed[1]


def sample_age_seconds(records: tuple[dict[str, object], ...], now: float) -> float:
    """Return the oldest sample age against the campaign clock."""
    oldest = min(_number(record["created_at"], "created_at") for record in records)
    return now - oldest
