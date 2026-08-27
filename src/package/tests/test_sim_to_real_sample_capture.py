"""Read-only synchronized sample acquisition contract.

A sample is a camera frame plus a joint state bound to the same wall-clock
reading. Duplicated snapshots, stale readings and camera/joint skew must fail
closed before any output is promoted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from so101_pusht_benchmark.sim_to_real.sample_capture import (
    CameraFrame,
    DuplicateFrameSource,
    JointState,
    StaticJointSource,
    acquire_samples,
    build_sample,
    load_policy_limits,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation


BENCHMARK = Path(__file__).resolve().parents[1]


def _load_fixture_samples(directory: Path) -> list[dict[str, Any]]:
    raw = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    snapshots = cast("dict[str, Any]", raw)["snapshots"]
    return cast("list[dict[str, Any]]", snapshots)


def test_build_sample_distinct_ids_and_timestamps() -> None:
    now = 1_000_000.0
    first = build_sample(
        CameraFrame(timestamp=now, digest="a" * 64),
        JointState(
            timestamp=now,
            body_degrees=(0.0, 0.1, 0.2, 0.3, 0.4),
            device_digest="b" * 64,
            calibration_digest="c" * 64,
        ),
        now=now,
        sample_id="sample-000",
        policy=load_policy_limits(max_age=0.2, max_skew=0.03),
    )
    second = build_sample(
        CameraFrame(timestamp=now + 0.01, digest="d" * 64),
        JointState(
            timestamp=now + 0.01,
            body_degrees=(0.1, 0.2, 0.3, 0.4, 0.5),
            device_digest="b" * 64,
            calibration_digest="c" * 64,
        ),
        now=now + 0.01,
        sample_id="sample-001",
        policy=load_policy_limits(max_age=0.2, max_skew=0.03),
    )
    assert first.record_id != second.record_id
    assert first.digest != second.digest
    assert second.created_at > first.created_at


def test_duplicate_snapshot_rejects_before_promotion() -> None:
    with pytest.raises(RolloutViolation) as exc_info:
        acquire_samples(
            DuplicateFrameSource(
                frames=[b"same", b"same"],
                base=1_000_000.0,
            ),
            StaticJointSource(
                states=[
                    JointState(
                        timestamp=1_000_000.0,
                        body_degrees=(0.0, 0.1, 0.2, 0.3, 0.4),
                        device_digest="b" * 64,
                        calibration_digest="c" * 64,
                    ),
                    JointState(
                        timestamp=1_000_000.01,
                        body_degrees=(0.0, 0.1, 0.2, 0.3, 0.4),
                        device_digest="b" * 64,
                        calibration_digest="c" * 64,
                    ),
                ],
                count=2,
            ),
            count=2,
            policy=load_policy_limits(max_age=0.2, max_skew=0.03),
            clock=iter([1_000_000.0, 1_000_000.01]).__next__,
        )
    assert exc_info.value.code is RolloutCode.R_DUPLICATE_SAMPLE


def test_skew_rejects() -> None:
    with pytest.raises(RolloutViolation) as exc_info:
        build_sample(
            CameraFrame(timestamp=1_000_000.0, digest="a" * 64),
            JointState(
                timestamp=1_000_000.1,
                body_degrees=(0.0, 0.1, 0.2, 0.3, 0.4),
                device_digest="b" * 64,
                calibration_digest="c" * 64,
            ),
            now=1_000_000.1,
            sample_id="sample-skewed",
            policy=load_policy_limits(max_age=0.2, max_skew=0.03),
        )
    assert exc_info.value.code is RolloutCode.R_STALE


def test_stale_rejects() -> None:
    with pytest.raises(RolloutViolation) as exc_info:
        build_sample(
            CameraFrame(timestamp=1_000_000.0, digest="a" * 64),
            JointState(
                timestamp=1_000_000.0,
                body_degrees=(0.0, 0.1, 0.2, 0.3, 0.4),
                device_digest="b" * 64,
                calibration_digest="c" * 64,
            ),
            now=1_000_000.3,
            sample_id="sample-stale",
            policy=load_policy_limits(max_age=0.2, max_skew=0.03),
        )
    assert exc_info.value.code is RolloutCode.R_STALE


def test_fixture_sample_stream_has_distinct_frames_and_states() -> None:
    samples = _load_fixture_samples(BENCHMARK / "tests/fixtures/sim_to_real/sample_stream")
    assert len(samples) == 2
    assert samples[0]["frame"] != samples[1]["frame"]
    assert samples[0]["body_degrees"] != samples[1]["body_degrees"]


def test_duplicate_fixture_points_to_same_snapshot() -> None:
    samples = _load_fixture_samples(
        BENCHMARK / "tests/fixtures/sim_to_real/duplicate_sample_stream"
    )
    assert len(samples) == 2
    assert samples[0]["frame"] == samples[1]["frame"]
    assert samples[0]["body_degrees"] == samples[1]["body_degrees"]
