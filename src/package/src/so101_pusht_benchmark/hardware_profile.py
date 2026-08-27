"""Strict loading and non-actuating validation for the real SO-101 setup."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, cast

import yaml

from so101_pusht_benchmark.sim_to_real.readiness import (
    bound_digest_blockers,
    is_sha256_digest,
    joint_mapping_blockers,
)


ArmRole = Literal["follower", "leader"]
EXPECTED_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True, slots=True)
class ArmProfile:
    role: ArmRole
    port: Path
    calibration_id: str
    calibration_file: Path


@dataclass(frozen=True, slots=True)
class CameraProfile:
    role: str
    device: Path
    width: int
    height: int
    fps: int
    crop_x: int
    crop_y: int
    crop_size: int
    saved_width: int
    saved_height: int
    latest_frame: Path


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    schema: int
    deployment_scope: str
    follower: ArmProfile
    leader: ArmProfile
    camera: CameraProfile
    max_relative_target_degrees: float
    require_workspace_confirmation: bool
    camera_registration_calibrated: bool
    action_bridge: str
    diagnostic_governance: str
    control_plane: str
    training_identity_authority: str
    lineage_digest: str
    policy_digest: str
    camera_registration_digest: str
    joint_equivalence_digest: str


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, object], value)


def _text(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _boolean(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be boolean")
    return value


def _number(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{key} must be a number")
    return float(value)


def _optional_digest(mapping: Mapping[str, object], key: str) -> str:
    if key not in mapping:
        return ""
    value = mapping[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _arm(mapping: Mapping[str, object], expected_role: ArmRole) -> ArmProfile:
    role = _text(mapping, "role")
    if role != expected_role:
        raise ValueError(f"expected {expected_role} arm, got {role}")
    return ArmProfile(
        role=expected_role,
        port=Path(_text(mapping, "port")),
        calibration_id=_text(mapping, "calibration_id"),
        calibration_file=Path(_text(mapping, "calibration_file")),
    )


def load_hardware_profile(path: Path) -> HardwareProfile:
    """Parse one real-hardware profile and reject ambiguous device roles."""
    raw = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "profile")
    safety = _mapping(raw.get("safety"), "safety")
    sim_to_real = _mapping(raw.get("sim_to_real"), "sim_to_real")
    camera = _mapping(raw.get("camera"), "camera")
    profile = HardwareProfile(
        schema=_integer(raw, "schema"),
        deployment_scope=_text(raw, "deployment_scope"),
        follower=_arm(_mapping(raw.get("follower"), "follower"), "follower"),
        leader=_arm(_mapping(raw.get("leader"), "leader"), "leader"),
        camera=CameraProfile(
            role=_text(camera, "role"),
            device=Path(_text(camera, "device")),
            width=_integer(camera, "width"),
            height=_integer(camera, "height"),
            fps=_integer(camera, "fps"),
            crop_x=_integer(camera, "crop_x"),
            crop_y=_integer(camera, "crop_y"),
            crop_size=_integer(camera, "crop_size"),
            saved_width=_integer(camera, "saved_width"),
            saved_height=_integer(camera, "saved_height"),
            latest_frame=Path(_text(camera, "latest_frame")),
        ),
        max_relative_target_degrees=_number(safety, "max_relative_target_degrees"),
        require_workspace_confirmation=_boolean(safety, "require_workspace_confirmation"),
        camera_registration_calibrated=_boolean(
            sim_to_real, "physical_camera_registration_calibrated"
        ),
        action_bridge=_text(sim_to_real, "action_bridge"),
        diagnostic_governance=_text(sim_to_real, "diagnostic_governance"),
        control_plane=_text(sim_to_real, "control_plane"),
        training_identity_authority=_text(sim_to_real, "training_identity_authority"),
        lineage_digest=_optional_digest(sim_to_real, "lineage_digest"),
        policy_digest=_optional_digest(sim_to_real, "policy_digest"),
        camera_registration_digest=_optional_digest(sim_to_real, "camera_registration_digest"),
        joint_equivalence_digest=_optional_digest(sim_to_real, "joint_equivalence_digest"),
    )
    if profile.schema != 1:
        raise ValueError(f"unsupported hardware profile schema {profile.schema}")
    if profile.deployment_scope != "real_hardware_preflight":
        raise ValueError("real hardware profile must remain preflight-only")
    if (
        profile.diagnostic_governance != "real_diagnostic_rollout"
        or profile.control_plane != "separate_control_plane"
        or profile.training_identity_authority != "forbidden"
    ):
        raise ValueError("simulation identity cannot control hardware")
    if profile.max_relative_target_degrees <= 0:
        raise ValueError("max_relative_target_degrees must be positive")
    return profile


def compatibility_blockers(profile: HardwareProfile) -> tuple[str, ...]:
    """Return blockers that prevent simulation checkpoints controlling hardware."""
    blockers: list[str] = []
    if not is_sha256_digest(profile.camera_registration_digest):
        blockers.append(
            "checkpoint expects simulator cam_top at 96x96; "
            "physical front crop is not registered to that view"
        )
    if not is_sha256_digest(profile.joint_equivalence_digest):
        blockers.append("checkpoint state is five simulated joints; follower exposes six motors")
    if profile.action_bridge != "audited":
        blockers.append(
            "checkpoint action is absolute_mocap_xy; no audited XY-to-joint bridge exists"
        )
    if not profile.camera_registration_calibrated:
        blockers.append(
            "physical camera intrinsics/extrinsics and table registration are not calibrated"
        )
    return tuple(blockers)


def rollout_readiness_blockers(
    profile: HardwareProfile,
    inference_receipt: Mapping[str, object],
    *,
    confirmed: bool = False,
) -> tuple[str, ...]:
    """Fail closed before any physical policy rollout can be considered."""
    blockers = list(compatibility_blockers(profile))
    if inference_receipt.get("mode") == "physical_frame_shadow_only":
        blockers.append("latest inference receipt is shadow-only")
    if inference_receipt.get("deployment_valid") is not True:
        blockers.append("latest inference receipt is not deployment-valid")
    if not confirmed:
        blockers.append("explicit low-speed rollout confirmation is absent")
    blockers.extend(
        bound_digest_blockers(profile.lineage_digest, inference_receipt, "lineage_digest")
    )
    blockers.extend(
        bound_digest_blockers(profile.policy_digest, inference_receipt, "policy_digest")
    )
    blockers.extend(
        bound_digest_blockers(
            profile.camera_registration_digest,
            inference_receipt,
            "camera_registration_digest",
        )
    )
    blockers.extend(joint_mapping_blockers(profile.joint_equivalence_digest, inference_receipt))
    if _elbow_mapping_invalid(inference_receipt):
        blockers.append("elbow mapping exceeds physical and MuJoCo ranges")
    return tuple(blockers)


def _elbow_mapping_invalid(receipt: Mapping[str, object]) -> bool:
    joints = receipt.get("joints")
    if not isinstance(joints, Mapping):
        return False
    elbow = _mapping(cast("Mapping[str, object]", joints), "joints").get("elbow_flex")
    if not isinstance(elbow, Mapping):
        return False
    return _mapping(cast("Mapping[str, object]", elbow), "elbow_flex").get("valid") is False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/hardware/so101_real_v1.yaml"),
    )
    args = parser.parse_args()
    from so101_pusht_benchmark.hardware_live import device_holders, live_checks

    profile = load_hardware_profile(args.profile.resolve())
    checks = live_checks(profile)
    hardware_ready = all(checks.values())
    blockers = compatibility_blockers(profile)
    report = {
        "profile": str(args.profile.resolve()),
        "hardware_ready": hardware_ready,
        "checks": checks,
        "camera_device_holders": device_holders(profile.camera.device),
        "sim_to_real_ready": not blockers,
        "sim_to_real_blockers": blockers,
        "safety": {
            "actuation_performed": False,
            "max_relative_target_degrees": profile.max_relative_target_degrees,
        },
    }
    print(json.dumps(report, indent=2))
    return 0 if hardware_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
