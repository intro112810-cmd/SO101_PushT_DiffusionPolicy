"""Fail-closed affine calibration between physical and MuJoCo joint ranges."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypedDict

from .contracts import validate_follower_receipt


ENCODER_MAX: Final = 4095
JOINT_ORDER: Final = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
MAPPING_FORMULA: Final = (
    "q_sim = q_min + (degree - degree_min) / (degree_max - degree_min) * (q_max - q_min)"
)


class JointMappingEntry(TypedDict):
    """JSON-compatible evidence for one physical-to-simulator joint mapping."""

    physical_degree: float
    physical_degree_range: list[float]
    mujoco_range_radians: list[float]
    mapped_q_radians: float
    physical_in_range: bool
    mujoco_in_range: bool
    valid: bool
    blockers: list[str]


class JointMappingReceipt(TypedDict):
    """JSON-compatible fail-closed receipt for one static read-only pose."""

    schema: int
    mode: str
    joint_order: list[str]
    mapping_formula: str
    joints: dict[str, JointMappingEntry]
    excluded_physical_motors: dict[str, str]
    mapping_status: str
    joint_frame_equivalence: str
    blockers: list[str]
    deployment_valid: bool
    clipping_performed: bool
    motor_writes_performed: bool
    actuation_performed: bool
    stop_boundary: str


def calibrated_degree_range(
    range_min: int,
    range_max: int,
) -> tuple[float, float]:
    """Match LeRobot's degree normalization at the calibrated encoder endpoints."""
    if isinstance(range_min, bool) or isinstance(range_max, bool):
        raise TypeError("encoder calibration endpoints must be integers")
    if range_min >= range_max:
        raise ValueError("encoder calibration range_min must be below range_max")
    midpoint = (range_min + range_max) / 2
    return (
        (range_min - midpoint) * 360 / ENCODER_MAX,
        (range_max - midpoint) * 360 / ENCODER_MAX,
    )


def affine_map_without_clipping(
    degree: float,
    *,
    degree_min: float,
    degree_max: float,
    q_min: float,
    q_max: float,
) -> float:
    """Apply the audited affine formula and preserve out-of-range results."""
    if degree_min >= degree_max:
        raise ValueError("physical degree_min must be below degree_max")
    if q_min >= q_max:
        raise ValueError("MuJoCo q_min must be below q_max")
    return q_min + (degree - degree_min) / (degree_max - degree_min) * (q_max - q_min)


def _calibration_range(
    calibration: Mapping[str, Mapping[str, int]],
    joint: str,
) -> tuple[float, float]:
    if joint not in calibration:
        raise ValueError(f"physical calibration is missing {joint}")
    endpoints = calibration[joint]
    if set(endpoints) < {"range_min", "range_max"}:
        raise ValueError(f"physical calibration for {joint} lacks range endpoints")
    return calibrated_degree_range(endpoints["range_min"], endpoints["range_max"])


def _joint_mapping(
    *,
    degree: float,
    degree_range: tuple[float, float],
    mujoco_range: tuple[float, float],
) -> JointMappingEntry:
    degree_min, degree_max = degree_range
    q_min, q_max = mujoco_range
    mapped_q = affine_map_without_clipping(
        degree,
        degree_min=degree_min,
        degree_max=degree_max,
        q_min=q_min,
        q_max=q_max,
    )
    physical_in_range = degree_min <= degree <= degree_max
    mujoco_in_range = q_min <= mapped_q <= q_max
    blockers: list[str] = []
    if not physical_in_range:
        blockers.append("physical_calibration_range_exceeded")
    if not mujoco_in_range:
        blockers.append("mujoco_joint_range_exceeded")
    return {
        "physical_degree": degree,
        "physical_degree_range": [degree_min, degree_max],
        "mujoco_range_radians": [q_min, q_max],
        "mapped_q_radians": mapped_q,
        "physical_in_range": physical_in_range,
        "mujoco_in_range": mujoco_in_range,
        "valid": not blockers,
        "blockers": blockers,
    }


def build_joint_mapping_receipt(
    *,
    calibration: Mapping[str, Mapping[str, int]],
    follower_receipt: object,
    mujoco_ranges: Mapping[str, tuple[float, float]],
) -> JointMappingReceipt:
    """Build non-deployable mapping evidence from one verified read-only pose."""
    if set(mujoco_ranges) != set(JOINT_ORDER):
        raise ValueError("MuJoCo ranges must contain exactly five mapped joints")
    positions = validate_follower_receipt(follower_receipt)
    joints: dict[str, JointMappingEntry] = {}
    blockers: list[str] = []
    for joint in JOINT_ORDER:
        entry = _joint_mapping(
            degree=positions[joint],
            degree_range=_calibration_range(calibration, joint),
            mujoco_range=mujoco_ranges[joint],
        )
        joints[joint] = entry
        blockers.extend(f"{joint}:{blocker}" for blocker in entry["blockers"])
    return {
        "schema": 1,
        "mode": "read_only_joint_mapping_calibration",
        "joint_order": list(JOINT_ORDER),
        "mapping_formula": MAPPING_FORMULA,
        "joints": joints,
        "excluded_physical_motors": {
            "gripper": "percentage space; excluded from the five-joint degree mapping"
        },
        "mapping_status": "invalid_blocked" if blockers else "provisional_static_pose_only",
        "joint_frame_equivalence": "not_audited_single_static_pose",
        "blockers": blockers,
        "deployment_valid": False,
        "clipping_performed": False,
        "motor_writes_performed": False,
        "actuation_performed": False,
        "stop_boundary": "diagnostic only; no physical command path exists",
    }
