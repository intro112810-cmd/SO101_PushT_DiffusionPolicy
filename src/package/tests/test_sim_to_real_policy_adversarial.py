from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from sim_to_real_policy_helpers import APPROVED, YamlValue, raw_policy, reject_policy, set_path
from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "policy_version",
        "policy_id",
        "approved_by",
        "approved_at",
        "valid_from",
        "expires_at",
        "canonical_digest",
        "owner_approval",
    ],
)
def test_every_arming_identity_and_approval_field_is_required(field: str, tmp_path: Path) -> None:
    raw = raw_policy()
    raw[field] = None
    reject_policy(raw, tmp_path / "missing.yaml", NOW)


def test_unapproved_status_and_malformed_yaml_reject(tmp_path: Path) -> None:
    raw = raw_policy()
    raw["approval_status"] = "pending"
    reject_policy(raw, tmp_path / "pending.yaml", NOW)
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("thresholds: [unterminated", encoding="utf-8")
    with pytest.raises(RolloutViolation) as caught:
        load_fixture_safety_policy(malformed, now=NOW)
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("thresholds", "timing", "sample_max_age_seconds"), None),
        (("thresholds", "camera"), {"min_correspondences": 12}),
        (("thresholds", "watchdog", "timeout_seconds"), 0),
        (("thresholds", "slew", "max_joint_delta_degrees"), -1.0),
        (("thresholds", "camera", "max_reprojection_error_px"), float("nan")),
        (("thresholds", "bounded_rollout", "max_duration_seconds"), float("inf")),
        (("thresholds", "single_step", "max_commands"), 2),
        (("thresholds", "camera", "min_correspondences"), 1.5),
        (("thresholds", "provider", "exact_goal_required"), "true"),
    ],
)
def test_missing_zero_negative_nonfinite_or_wrong_domain_rejects(
    path: tuple[str, ...], value: YamlValue, tmp_path: Path
) -> None:
    raw = raw_policy()
    set_path(raw, path, value)
    reject_policy(raw, tmp_path / "numeric.yaml", NOW)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (
            ("thresholds", "joint_domains", "physical_degrees", "elbow_flex"),
            [105.0, -105.0],
        ),
        (("thresholds", "timing", "sample_max_skew_seconds"), 0.3),
        (("thresholds", "timing", "authorization_ttl_seconds"), 11.0),
        (("thresholds", "camera", "max_reprojection_error_px"), 3.0),
        (("thresholds", "acknowledgement", "timeout_seconds"), 0.3),
    ],
)
def test_inconsistent_ranges_and_cross_thresholds_reject(
    path: tuple[str, ...], value: YamlValue, tmp_path: Path
) -> None:
    raw = raw_policy()
    set_path(raw, path, value)
    reject_policy(raw, tmp_path / "inconsistent.yaml", NOW)


def test_digest_drift_rejects_semantic_and_metadata_mutation(tmp_path: Path) -> None:
    for index, (path, value) in enumerate(
        (
            (("thresholds", "watchdog", "timeout_seconds"), 0.3),
            (("approved_at",), "2026-08-23T00:00:01Z"),
            (("approved_by",), "other@example.invalid"),
        )
    ):
        raw = raw_policy()
        set_path(raw, path, value)
        reject_policy(raw, tmp_path / f"drift-{index}.yaml", NOW)


@pytest.mark.parametrize(
    ("field", "value", "now"),
    [
        ("approved_at", "2025-01-01T00:00:00Z", datetime(2026, 8, 23, tzinfo=timezone.utc)),
        ("approved_at", "2026-08-24T00:00:00Z", datetime(2026, 8, 23, tzinfo=timezone.utc)),
        ("expires_at", "2026-08-23T01:00:00Z", datetime(2026, 8, 23, 2, tzinfo=timezone.utc)),
        ("valid_from", "2026-08-24T00:00:00Z", datetime(2026, 8, 23, tzinfo=timezone.utc)),
    ],
)
def test_stale_future_or_expired_policy_rejects(
    field: str, value: str, now: datetime, tmp_path: Path
) -> None:
    raw = raw_policy()
    raw[field] = value
    reject_policy(raw, tmp_path / "temporal.yaml", now)


def test_unknown_and_duplicate_yaml_key_reject(tmp_path: Path) -> None:
    raw = raw_policy()
    raw["executor_override"] = True
    reject_policy(raw, tmp_path / "unknown.yaml", NOW)
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(APPROVED.read_text(encoding="utf-8") + "policy_id: duplicate\n")
    with pytest.raises(RolloutViolation) as caught:
        load_fixture_safety_policy(duplicate, now=NOW)
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED


@pytest.mark.parametrize(
    "polygon",
    [
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
    ],
)
def test_malformed_polygon_rejects(polygon: YamlValue, tmp_path: Path) -> None:
    raw = raw_policy()
    set_path(raw, ("thresholds", "workspace", "polygon_xy_m"), polygon)
    reject_policy(raw, tmp_path / "polygon.yaml", NOW)


def test_unbound_approval_mutations_reject(tmp_path: Path) -> None:
    for index, (field, value) in enumerate(
        (
            ("signer_id", "attacker@example.invalid"),
            ("policy_digest", "0" * 64),
            ("binding_signature", "00" * 256),
            ("approval_id", "other-approval"),
        )
    ):
        raw = raw_policy()
        set_path(raw, ("owner_approval", field), value)
        reject_policy(raw, tmp_path / f"approval-{index}.yaml", NOW)
