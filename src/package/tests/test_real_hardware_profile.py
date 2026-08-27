from __future__ import annotations

from pathlib import Path

from so101_pusht_benchmark.hardware_profile import (
    compatibility_blockers,
    load_hardware_profile,
    rollout_readiness_blockers,
)
from so101_pusht_benchmark.workspace import load_workspace_policy


ROOT = Path(__file__).parents[1]
PROFILE = ROOT / "configs/hardware/so101_real_v1.yaml"
READY_PROFILE = ROOT / "tests/fixtures/sim_to_real/ready_profile.yaml"
DEPLOYMENT_VALID = ROOT / "tests/fixtures/sim_to_real/deployment_valid.json"


def test_real_hardware_profile_uses_stable_calibrated_devices() -> None:
    profile = load_hardware_profile(PROFILE)

    assert profile.follower.role == "follower"
    assert profile.follower.port.name.endswith("5AE6082660-if00")
    assert profile.follower.calibration_id == "intro_so101_follower_01"
    assert profile.leader.role == "leader"
    assert profile.leader.port.name.endswith("5AE6082503-if00")
    assert profile.leader.calibration_id == "intro_so101_leader_01"
    assert profile.camera.device.name.endswith("SN0001-video-index0")
    assert (
        profile.camera.crop_x,
        profile.camera.crop_y,
        profile.camera.crop_size,
    ) == (100, 0, 400)
    assert (profile.camera.saved_width, profile.camera.saved_height) == (400, 400)
    assert profile.max_relative_target_degrees == 5.0


def test_training_identity_remains_simulation_only() -> None:
    policy = load_workspace_policy()
    profile = load_hardware_profile(PROFILE)

    identities = policy["model_authority"]["identities"]
    assert set(identities) == {"dp_cnn", "dp_transformer", "ibc", "lstm_gmm"}
    assert all(set(scopes.values()) == {"simulation_only"} for scopes in identities.values())
    assert profile.diagnostic_governance == "real_diagnostic_rollout"
    assert profile.training_identity_authority == "forbidden"


def test_simulation_checkpoint_mismatches_are_explicit() -> None:
    profile = load_hardware_profile(PROFILE)

    assert compatibility_blockers(profile) == (
        (
            "checkpoint expects simulator cam_top at 96x96; "
            "physical front crop is not registered to that view"
        ),
        "checkpoint state is five simulated joints; follower exposes six motors",
        "checkpoint action is absolute_mocap_xy; no audited XY-to-joint bridge exists",
        "physical camera intrinsics/extrinsics and table registration are not calibrated",
    )


def test_rollout_gate_rejects_shadow_only_receipt() -> None:
    profile = load_hardware_profile(PROFILE)

    blockers = rollout_readiness_blockers(
        profile,
        {
            "mode": "physical_frame_shadow_only",
            "actuation_performed": False,
            "deployment_valid": False,
        },
    )

    assert "latest inference receipt is shadow-only" in blockers
    assert "explicit low-speed rollout confirmation is absent" in blockers


def test_boolean_flags_cannot_promote_rollout() -> None:
    """Booleans and explicit confirmation never replace content-addressed evidence."""
    profile = load_hardware_profile(READY_PROFILE)

    assert profile.camera_registration_calibrated is True
    assert profile.action_bridge == "audited"

    blockers = rollout_readiness_blockers(
        profile,
        {
            "mode": "sim_to_real_deployment_ready",
            "deployment_valid": True,
        },
        confirmed=True,
    )

    assert blockers
    assert "explicit low-speed rollout confirmation is absent" not in blockers
    assert "lineage evidence is missing its content digest" in blockers
    assert "policy evidence is missing its content digest" in blockers
    assert "camera registration evidence is missing its receipt hash" in blockers
    assert "joint mapping evidence is missing a valid receipt status" in blockers


def test_ready_fixture_is_content_addressed() -> None:
    """The ready fixture promotes only when every bound evidence hash matches."""
    import json

    profile = load_hardware_profile(READY_PROFILE)
    receipt = json.loads(DEPLOYMENT_VALID.read_text(encoding="utf-8"))

    blockers = rollout_readiness_blockers(
        profile,
        receipt,
        confirmed=True,
    )

    assert blockers == ()
