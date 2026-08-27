"""Fail-closed observation, action, and timing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import cast, TYPE_CHECKING

import numpy as np

from ..task.spec import EE_Z_BOUNDS_M

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Mapping

__all__ = [
    "ContractError",
    "FrontRGB",
    "Observation",
    "PolicyInput",
    "TimingContract",
    "TopdownRGB",
]


class ContractError(ValueError):
    """Raised when a boundary value violates the frozen contract."""


def _array(
    value: object, shape: tuple[int, ...], dtype: np.dtype[np.generic]
) -> NDArray[np.generic]:
    if type(value) is not np.ndarray:
        raise ContractError("value must be an exact numpy.ndarray")
    array = cast("NDArray[np.generic]", value)
    if array.shape != shape:
        raise ContractError(f"expected shape {shape}")
    if array.dtype != dtype:
        raise ContractError(f"value must be {dtype}")
    if not bool(np.isfinite(array).all()):
        raise ContractError("value contains NaN or infinity")
    return array


def _state_values(array: NDArray[np.generic]) -> tuple[float, ...]:
    values = cast("list[float]", array.tolist())
    return tuple(float(value) for value in values)


@dataclass(frozen=True, slots=True)
class FrontRGB:
    """Front camera frame metadata and immutable pixel payload."""

    data: NDArray[np.generic]
    shape: tuple[int, int, int] = (96, 96, 3)
    dtype: str = "uint8"

    @classmethod
    def parse(cls, value: object) -> FrontRGB:
        return cls(_array(value, (96, 96, 3), np.dtype(np.uint8)))


@dataclass(frozen=True, slots=True)
class TopdownRGB:
    """Topdown camera frame metadata and immutable pixel payload."""

    data: NDArray[np.generic]
    shape: tuple[int, int, int] = (96, 96, 3)
    dtype: str = "uint8"

    @classmethod
    def parse(cls, value: object) -> TopdownRGB:
        return cls(_array(value, (96, 96, 3), np.dtype(np.uint8)))


@dataclass(frozen=True, slots=True)
class Observation:
    """Policy observation; telemetry is intentionally not representable here."""

    JOINT_NAMES = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    front: FrontRGB | None
    state: tuple[float, ...]
    topdown: TopdownRGB | None = None
    state_dtype: str = "float32"
    state_units: tuple[str, str, str] = ("radians", "radians_per_second", "metres")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.JOINT_NAMES

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> Observation:
        keys = set(value)
        if keys == {"observation.images.front", "observation.state"}:
            front = FrontRGB.parse(value["observation.images.front"])
            state = _array(value["observation.state"], (15,), np.dtype(np.float32))
            return cls(front=front, state=_state_values(state), topdown=None)
        if keys == {"observation.images.topdown", "observation.state"}:
            topdown = TopdownRGB.parse(value["observation.images.topdown"])
            state = _array(value["observation.state"], (15,), np.dtype(np.float32))
            return cls(front=None, state=_state_values(state), topdown=topdown)
        raise ContractError("observation keys must be exactly the policy allowlist")


@dataclass(frozen=True, slots=True)
class PolicyInput:
    """Absolute end-effector XYZ action in task-frame metres."""

    action: tuple[float, float, float]
    dtype: str = "float32"
    units: str = "metres"
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.18, 0.38),
        (-0.16, 0.16),
        EE_Z_BOUNDS_M,
    )

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> PolicyInput:
        if set(value) != {"action"}:
            raise ContractError("policy keys must be exactly {'action'}; telemetry is excluded")
        array = _array(value["action"], (3,), np.dtype(np.float32))
        action = _state_values(array)
        bounds = ((0.18, 0.38), (-0.16, 0.16), EE_Z_BOUNDS_M)
        if not all(
            np.float32(low) <= np.float32(item) <= np.float32(high)
            for item, (low, high) in zip(action, bounds, strict=True)
        ):
            raise ContractError("action is outside the configured XYZ envelope")
        return cls((action[0], action[1], action[2]))


@dataclass(frozen=True, slots=True)
class TimingContract:
    """One 10 Hz tick: observation first, action held for 50 physics steps."""

    frame_index: int
    timestamp: float
    fps: int = 10
    mujoco_dt: float = 0.002
    substeps: int = 50

    @classmethod
    def create(cls, frame_index: int, timestamp: float) -> TimingContract:
        if type(frame_index) is not int or frame_index < 0 or not isfinite(timestamp):
            raise ContractError("frame_index must be a non-bool integral and timestamp finite")
        expected = frame_index / 10
        if abs(timestamp - expected) > 1e-12:
            raise ContractError("timestamp must equal frame_index / 10")
        return cls(frame_index, timestamp)

    @property
    def action_interval(self) -> tuple[float, float]:
        return (self.timestamp, self.timestamp + self.substeps * self.mujoco_dt)

    def validate_next(self, other: TimingContract) -> None:
        if (
            other.frame_index != self.frame_index + 1
            or abs(other.timestamp - (self.timestamp + 0.1)) > 1e-12
        ):
            raise ContractError("timing frames must be contiguous at 10 Hz")
