"""Strict, immutable Push-T task and experiment specification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from .validation import (
    SAFETY_FIELDS,
    SpecError,
    Quotas,
    ResetDistribution,
    exact,
    integer,
    models,
    number,
    mapping,
    quotas,
    reset,
    seeds,
)

__all__ = [
    "APPROACH_HEIGHT_M",
    "CONTACT_HEIGHT_M",
    "EE_Z_BOUNDS_M",
    "TaskSpec",
    "TaskSpecError",
]

# Numeric source: frozen ``configs/calibration/sim_envelope_v1.yaml`` records the
# measured contact height as 0.045 m. No frozen approach-height calibration was
# available, so the approved-plan default safe approach height is 0.050 m.
# The robot-direct-control plan widens the z envelope: the kinematic arm can
# hold any z above the table (top 0.015 + sphere 0.012 + margin) up to a safe
# reach ceiling, and the operator adjusts z freely with the keyboard.
CONTACT_HEIGHT_M = 0.045
APPROACH_HEIGHT_M = 0.050
EE_Z_MIN_M = 0.030
EE_Z_MAX_M = 0.100
EE_Z_BOUNDS_M = (EE_Z_MIN_M, EE_Z_MAX_M)

TaskSpecError = SpecError


def _validate_policy_schema(raw: Mapping[str, object], schema: int) -> None:
    observation = mapping(raw["observation"], "observation")
    exact(observation, {"image", "state", "joint_order"}, "observation")
    image = mapping(observation["image"], "observation.image")
    exact(image, {"key", "dtype", "shape", "layout"}, "observation.image")
    state = mapping(observation["state"], "observation.state")
    exact(state, {"key", "dtype", "shape", "units"}, "observation.state")

    if schema == 1:
        expected_image_key = "observation.images.front"
        expected_allowlist = ["observation.images.front", "observation.state", "action"]
        expected_z_bounds = [CONTACT_HEIGHT_M, APPROACH_HEIGHT_M]
    elif schema == 3:
        expected_image_key = "observation.images.topdown"
        expected_allowlist = ["observation.images.topdown", "observation.state", "action"]
        expected_z_bounds = [EE_Z_MIN_M, EE_Z_MAX_M]
    else:
        raise SpecError(f"unsupported schema {schema}")

    if image != {
        "key": expected_image_key,
        "dtype": "uint8",
        "shape": [96, 96, 3],
        "layout": "HWC",
    }:
        raise SpecError("image schema is invalid")
    if state != {
        "key": "observation.state",
        "dtype": "float32",
        "shape": [15],
        "units": ["radians", "radians_per_second", "metres"],
    }:
        raise SpecError("state schema is invalid")
    action = mapping(raw["action"], "action")
    exact(
        action,
        {"key", "dtype", "shape", "units", "meaning", "bounds", "controller_owned"},
        "action",
    )
    bounds = mapping(action["bounds"], "action.bounds")
    exact(bounds, {"x", "y", "z"}, "action.bounds")
    if (
        action["key"] != "action"
        or action["dtype"] != "float32"
        or action["shape"] != [3]
        or action["units"] != "meters"
        or action["meaning"] != "absolute_ee_xyz_target_task_frame"
        or bounds
        != {
            "x": [0.18, 0.38],
            "y": [-0.16, 0.16],
            "z": expected_z_bounds,
        }
    ):
        raise SpecError("action schema is invalid")
    if tuple(cast("list[str]", action["controller_owned"])) != ("orientation", "gripper_open"):
        raise SpecError("controller-owned action fields are invalid")
    if raw["policy_allowlist"] != expected_allowlist:
        raise SpecError("policy allowlist is invalid")
    if schema == 1:
        expected_telemetry = [
            "raw_gamepad",
            "requested_target",
            "ik_target",
            "driver_command",
            "measured_next_state",
            "coverage",
            "ik_diagnostics",
            "timestamp",
        ]
    else:
        expected_telemetry = [
            "raw_mouse_keyboard",
            "requested_target",
            "ik_target",
            "driver_command",
            "measured_next_state",
            "coverage",
            "ik_diagnostics",
            "timestamp",
        ]
    if raw["telemetry_schema"] != expected_telemetry:
        raise SpecError("telemetry schema is invalid")


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Fully parsed benchmark contract with no implicit coercions."""

    schema: int
    identifier: str
    horizon: int
    success_coverage: float
    reset: ResetDistribution
    quotas: Quotas
    deployment_scope: str
    target_pose: tuple[float, float, float]
    policy_fps: int
    mujoco_dt: float
    substeps: int
    evaluation_seeds: tuple[int, ...]
    training_updates: int
    models: Mapping[str, tuple[int, int, int]]
    safety_envelope: Mapping[str, object]

    @classmethod
    def parse(cls, raw: Mapping[str, object]) -> TaskSpec:
        """Parse exactly the versioned machine-readable schema."""
        expected = {
            "schema",
            "identifier",
            "deployment_scope",
            "policy_fps",
            "mujoco_dt",
            "substeps",
            "horizon",
            "success_coverage",
            "target_pose",
            "reset",
            "observation",
            "action",
            "policy_allowlist",
            "telemetry_schema",
            "safety_envelope",
            "quotas",
            "data",
            "model_protocol",
        }
        exact(raw, expected, "task")
        if raw["schema"] not in (1, 3) or raw["deployment_scope"] != "simulation_only":
            raise SpecError("schema or deployment scope is invalid")
        identifier = raw["identifier"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or Path(identifier).is_absolute()
            or ".." in Path(identifier).parts
        ):
            raise SpecError("unsafe task identifier")
        horizon = integer(raw["horizon"], "horizon")
        if not 1 <= horizon <= 300:
            raise SpecError("horizon must be in [1, 300]")
        coverage = number(raw["success_coverage"], "success_coverage")
        if not 0 <= coverage <= 1:
            raise SpecError("success coverage must be in [0, 1]")
        if (raw["policy_fps"], raw["substeps"], raw["mujoco_dt"]) != (10, 50, 0.002):
            raise SpecError("timing constants are not the locked contract")
        _validate_policy_schema(raw, cast(int, raw["schema"]))
        data = mapping(raw["data"], "data")
        exact(
            data,
            {
                "pilot_successful_episodes",
                "accepted_demonstrations",
                "splits",
                "training_seeds",
                "evaluation_seeds",
            },
            "data",
        )
        if (data["accepted_demonstrations"], data["pilot_successful_episodes"]) != (200, 20):
            raise SpecError("demo quantities are not locked")
        splits = mapping(data["splits"], "splits")
        exact(splits, {"train", "validation", "test"}, "splits")
        if dict(splits) != {"train": 160, "validation": 20, "test": 20}:
            raise SpecError("split quotas are invalid")
        training = seeds(data["training_seeds"], "training_seeds")
        evaluation = seeds(data["evaluation_seeds"], "evaluation_seeds")
        if evaluation != tuple(range(100000, 100100)) or set(training) & set(evaluation):
            raise SpecError("evaluation seeds must be exactly 100000..100099 and disjoint")
        pose_raw = raw["target_pose"]
        if not isinstance(pose_raw, list):
            raise SpecError("target_pose must be [x, y, yaw]")
        pose = cast("list[object]", pose_raw)
        if len(pose) != 3:
            raise SpecError("target_pose must be [x, y, yaw]")
        safety = mapping(raw["safety_envelope"], "safety_envelope")
        expected_safety: set[str] = set(SAFETY_FIELDS)
        if raw["schema"] == 3:
            expected_safety.add("clearance_z_m")
        exact(safety, expected_safety, "safety_envelope")
        parsed_models, updates = models(mapping(raw["model_protocol"], "model_protocol"))
        return cls(
            cast(int, raw["schema"]),
            identifier,
            horizon,
            coverage,
            reset(mapping(raw["reset"], "reset")),
            quotas(mapping(raw["quotas"], "quotas")),
            "simulation_only",
            (
                number(pose[0], "target_pose"),
                number(pose[1], "target_pose"),
                number(pose[2], "target_pose"),
            ),
            10,
            0.002,
            50,
            evaluation,
            updates,
            parsed_models,
            safety,
        )

    def require_safety_ready(self) -> None:
        """Reject use until every required calibration value is numeric."""
        missing = tuple(key for key in SAFETY_FIELDS if self.safety_envelope[key] is None)
        if missing:
            raise SpecError("safety calibration missing: " + ", ".join(missing))

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskSpec:
        """Load and parse one YAML contract file."""
        try:
            parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise SpecError("cannot read task config") from exc
        if not isinstance(parsed, dict):
            raise SpecError("task config must be a mapping")
        return cls.parse(cast("Mapping[str, object]", parsed))
