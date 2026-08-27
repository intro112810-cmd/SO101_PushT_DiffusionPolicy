"""Typed validation helpers for the benchmark task schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast


class SpecError(ValueError):
    """Raised for malformed task contract values."""


STRATA = (
    "initial_yaw_negative_first_contact_left",
    "initial_yaw_negative_first_contact_right",
    "initial_yaw_positive_first_contact_left",
    "initial_yaw_positive_first_contact_right",
)
MODEL_CONTRACT = {
    "DP-CNN": (2, 16, 8),
    "DP-Transformer": (2, 16, 8),
    "IBC": (2, 2, 1),
    "LSTM-GMM": (10, 10, 1),
}
__all__ = ["Quotas", "ResetDistribution", "SpecError", "models", "quotas", "reset", "seeds"]

SAFETY_FIELDS = (
    "contact_z_m",
    "max_ee_step_m",
    "joint_soft_limit_margin_rad",
    "joint_command_delta_rad",
    "joint_velocity_rad_s",
    "ik_residual_m",
    "ik_iteration_cap",
    "contact_force_n",
    "contact_impulse_ns",
    "actuator_limits",
)


def mapping(raw: object, key: str) -> Mapping[str, object]:
    if not isinstance(raw, dict):
        raise SpecError(f"{key} must be a mapping")
    return cast("Mapping[str, object]", raw)


def exact(raw: Mapping[str, object], expected: set[str], key: str) -> None:
    unknown, missing = set(raw) - expected, expected - set(raw)
    if unknown:
        raise SpecError(f"{key} unknown keys: {sorted(unknown)}")
    if missing:
        raise SpecError(f"{key} missing keys: {sorted(missing)}")


def number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{key} must be numeric without coercion")
    return float(value)


def integer(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{key} must be an integer without coercion")
    return value


def value_range(raw: Mapping[str, object], key: str) -> tuple[float, float]:
    value = raw[key]
    if not isinstance(value, list):
        raise SpecError(f"{key} must be a two-item range")
    values = cast("list[object]", value)
    if len(values) != 2:
        raise SpecError(f"{key} must be a two-item range")
    result = (number(values[0], key), number(values[1], key))
    if result[0] > result[1]:
        raise SpecError(f"{key} is reversed")
    return result


@dataclass(frozen=True, slots=True)
class ResetDistribution:
    """Development and final reset distributions."""

    block_x: tuple[float, float]
    block_y: tuple[float, float]
    final_yaw: tuple[float, float]
    ee_x: tuple[float, float]
    ee_y: tuple[float, float]
    development_yaw: float
    min_block_clearance: float
    max_attempts: int


@dataclass(frozen=True, slots=True)
class Quotas:
    """Frozen dataset split quotas and strata."""

    train: int
    validation: int
    test: int
    strata: tuple[str, ...]

    @property
    def total(self) -> int:
        """Return total accepted demonstrations."""
        return self.train + self.validation + self.test


def reset(raw: Mapping[str, object]) -> ResetDistribution:
    exact(
        raw,
        {
            "development_yaw",
            "block_x",
            "block_y",
            "final_yaw",
            "ee_x",
            "ee_y",
            "min_block_clearance",
            "max_reset_attempts",
        },
        "reset",
    )
    development = number(raw["development_yaw"], "development_yaw")
    if development != 0.0 or value_range(raw, "final_yaw") != (
        -1.5707963267948966,
        1.5707963267948966,
    ):
        raise SpecError("development/final yaw reset distributions are invalid")
    attempts = integer(raw["max_reset_attempts"], "max_reset_attempts")
    clearance = number(raw["min_block_clearance"], "min_block_clearance")
    if attempts < 1 or clearance != 0.08:
        raise SpecError("reset attempts or clearance is invalid")
    return ResetDistribution(
        value_range(raw, "block_x"),
        value_range(raw, "block_y"),
        value_range(raw, "final_yaw"),
        value_range(raw, "ee_x"),
        value_range(raw, "ee_y"),
        development,
        clearance,
        attempts,
    )


def quotas(raw: Mapping[str, object]) -> Quotas:
    exact(raw, {"train", "validation", "test", "strata"}, "quotas")
    values = tuple(integer(raw[key], key) for key in ("train", "validation", "test"))
    strata_raw = raw["strata"]
    if not isinstance(strata_raw, list) or tuple(cast("list[str]", strata_raw)) != STRATA:
        raise SpecError("strata semantics are invalid")
    result = Quotas(values[0], values[1], values[2], STRATA)
    if result.total != 200:
        raise SpecError("quotas must total 200")
    return result


def seeds(raw: object, key: str) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise SpecError(f"{key} must contain integral seeds")
    values = cast("list[object]", raw)
    if not all(type(value) is int for value in values):
        raise SpecError(f"{key} must contain integral seeds")
    result = tuple(cast("list[int]", values))
    if len(set(result)) != len(result):
        raise SpecError(f"{key} contains duplicate seeds")
    return result


def models(raw: Mapping[str, object]) -> tuple[Mapping[str, tuple[int, int, int]], int]:
    exact(raw, {"batch_size", "max_train_steps", "num_epochs", "models"}, "model_protocol")
    if (raw["batch_size"], raw["max_train_steps"], raw["num_epochs"]) != (64, 5000, 20):
        raise SpecError("training budget is invalid")
    model_raw = mapping(raw["models"], "models")
    if set(model_raw) != set(MODEL_CONTRACT):
        raise SpecError("model declarations are incomplete")
    result: dict[str, tuple[int, int, int]] = {}
    for name, expected in MODEL_CONTRACT.items():
        values = mapping(model_raw[name], f"model:{name}")
        keys = {"sequence_length", "observation_steps", "prediction_horizon", "executed_actions"}
        omitted = "sequence_length" if name != "LSTM-GMM" else "observation_steps"
        if set(values) != keys - {omitted}:
            raise SpecError(f"model:{name} declaration keys are invalid")
        first = "sequence_length" if name == "LSTM-GMM" else "observation_steps"
        actual = (
            integer(values[first], first),
            integer(values["prediction_horizon"], "prediction_horizon"),
            integer(values["executed_actions"], "executed_actions"),
        )
        if actual != expected:
            raise SpecError(f"model:{name} horizon contract is invalid")
        result[name] = actual
    return result, 100_000
