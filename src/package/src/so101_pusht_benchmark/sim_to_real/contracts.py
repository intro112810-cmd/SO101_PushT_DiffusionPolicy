"""Strict receipt contracts for a no-actuation sim-to-real dry-run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import cast


MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class ContractError(ValueError):
    """A receipt cannot prove the required read-only boundary."""


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a mapping")
    return cast("Mapping[str, object]", value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be a finite number")
    return result


def _false(receipt: Mapping[str, object], key: str, label: str) -> None:
    if receipt.get(key) is not False:
        raise ContractError(f"{label} is not verified read-only: {key} must be false")


def validate_follower_receipt(receipt: object) -> dict[str, float]:
    """Return exact six-motor degrees only from a verified read-only receipt."""
    raw = _mapping(receipt, "follower receipt")
    if raw.get("mode") != "read_only_follower_state":
        raise ContractError("follower receipt mode is not read-only")
    _false(raw, "motor_writes_performed", "follower receipt")
    _false(raw, "actuation_performed", "follower receipt")
    positions = _mapping(raw.get("positions_degrees"), "positions_degrees")
    if set(positions) != set(MOTORS):
        raise ContractError("positions_degrees must contain exactly six motors")
    return {motor: _number(positions[motor], motor) for motor in MOTORS}


def validate_shadow_receipt(receipt: object) -> tuple[list[float], list[list[float]]]:
    """Return state and actions only from the immutable shadow-inference contract."""
    raw = _mapping(receipt, "shadow receipt")
    if raw.get("mode") != "physical_frame_shadow_only":
        raise ContractError("shadow receipt mode is invalid")
    _false(raw, "deployment_valid", "shadow receipt")
    _false(raw, "actuation_performed", "shadow receipt")
    _false(raw, "follower_motor_writes_performed", "shadow receipt")
    _false(raw, "follower_actuation_performed", "shadow receipt")
    if raw.get("checkpoint_image_contract") != "CCW90 RGB uint8[96,96,3]":
        raise ContractError("shadow image contract is not the physical CCW90 contract")
    state_raw = raw.get("agent_pos")
    if not isinstance(state_raw, Sequence) or isinstance(state_raw, (str, bytes)):
        raise ContractError("agent_pos must be a five-value sequence")
    state_values = cast("Sequence[object]", state_raw)
    state = [_number(value, "agent_pos") for value in state_values]
    if len(state) != 5:
        raise ContractError("agent_pos must contain exactly five values")
    actions_raw = raw.get("predicted_actions")
    if not isinstance(actions_raw, Sequence) or isinstance(actions_raw, (str, bytes)):
        raise ContractError("predicted_actions must be a sequence")
    actions: list[list[float]] = []
    action_values = cast("Sequence[object]", actions_raw)
    for index, action_raw in enumerate(action_values):
        if not isinstance(action_raw, Sequence) or isinstance(action_raw, (str, bytes)):
            raise ContractError(f"predicted action {index} must be a sequence")
        action_sequence = cast("Sequence[object]", action_raw)
        action = [_number(value, f"predicted action {index}") for value in action_sequence]
        if len(action) != 2 or any(value < -1.0 or value > 1.0 for value in action):
            raise ContractError(f"predicted action {index} violates float32[2] bounds")
        actions.append(action)
    if len(actions) != 8:
        raise ContractError("DP-CNN dry-run must contain exactly eight predicted actions")
    return state, actions


def build_dry_run_contract(
    follower_receipt: object,
    shadow_receipt: object,
) -> dict[str, object]:
    """Build the final immutable receipt without exposing an execution path."""
    positions = validate_follower_receipt(follower_receipt)
    shadow = _mapping(shadow_receipt, "shadow receipt")
    state, actions = validate_shadow_receipt(shadow)
    return {
        "schema": 1,
        "mode": "sim_to_real_dry_run",
        "model": shadow.get("model"),
        "artifact_id": shadow.get("artifact_id"),
        "evidence_scope": shadow.get("evidence_scope"),
        "policy_evidence": shadow.get("policy_evidence"),
        "frame_sha256": shadow.get("frame_sha256"),
        "checkpoint_image_contract": shadow.get("checkpoint_image_contract"),
        "physical_positions_degrees": positions,
        "agent_pos": state,
        "agent_pos_source": shadow.get("agent_pos_source"),
        "predicted_actions": actions,
        "preview_scope": "mujoco_trajectory_only",
        "mapping_status": "provisional_not_calibrated",
        "deployment_valid": False,
        "motor_writes_performed": False,
        "actuation_performed": False,
        "stop_boundary": "no physical command path exists",
    }
