from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sim_to_real_policy_helpers import APPROVED
from so101_pusht_benchmark.hardware_profile import load_hardware_profile
from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.policy_types import FixtureApprovedSafetyPolicy
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

ROOT = Path(__file__).parents[1]
PROFILE = ROOT / "configs/hardware/so101_real_v1.yaml"
PENDING = ROOT / "configs/hardware/sim_to_real_safety_policy_v1.pending.yaml"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
EXPECTED_DIGEST = "54f41dcc964169459dc4d77f64bfa4f53bcc21e2d405931cd0eb51f41af11a6a"


def test_baseline_legacy_profile_threshold_is_outside_sim_to_real_modules() -> None:
    profile = load_hardware_profile(PROFILE)

    assert profile.max_relative_target_degrees == 5.0
    assert not hasattr(profile, "approved_safety_policy")
    sim_to_real = ROOT / "src/so101_pusht_benchmark/sim_to_real"
    assert all(
        "max_relative_target_degrees" not in path.read_text(encoding="utf-8")
        for path in sim_to_real.glob("*.py")
    )


def test_exact_full_approved_fixture_parses_to_immutable_fixture_type() -> None:
    policy = load_fixture_safety_policy(APPROVED, now=NOW)

    assert type(policy) is FixtureApprovedSafetyPolicy
    assert policy.canonical_digest == EXPECTED_DIGEST
    assert policy.policy_id == "TEST-FIXTURE-ONLY-sim-to-real-safety-v1"
    assert policy.artifact_scope == "test_fixture_only"
    assert policy.workspace.polygon_xy_m == (
        (-0.3, -0.2),
        (0.3, -0.2),
        (0.3, 0.2),
        (-0.3, 0.2),
    )
    assert policy.joint_domains.joint_order == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    assert policy.joint_domains.physical_degrees[2].maximum == 105.0
    assert policy.joint_domains.mapped_radians[2].maximum == 1.69
    assert policy.timing.sample_max_age_seconds == 0.2
    assert policy.camera.min_correspondences == 12
    assert policy.kinematics.max_fk_residual_m == 0.003
    assert policy.slew.max_joint_delta_degrees == 3.0
    assert policy.provider.exact_goal_required is True
    assert policy.watchdog.timeout_seconds == 0.25
    assert policy.acknowledgement.required is True
    assert policy.post_state.max_tracking_error_degrees == 1.0
    assert policy.shadow.min_cycles == 100
    assert policy.single_step.max_commands == 1
    assert policy.bounded_rollout.max_commands == 10
    assert policy.operator.stop_behavior == "latch_hold"
    with pytest.raises(FrozenInstanceError):
        policy.__setattr__("policy_id", "mutated")


def test_pending_policy_is_rejected() -> None:
    with pytest.raises(RolloutViolation) as caught:
        load_fixture_safety_policy(PENDING, now=NOW)
    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED
    assert "policy is not owner-approved" in str(caught.value)
