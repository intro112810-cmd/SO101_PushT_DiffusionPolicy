"""Non-actuating adapters for physical-frame policy diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray


UInt8Image = NDArray[np.uint8]
Float32Vector = NDArray[np.float32]
PHYSICAL_MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SIMULATOR_JOINT_ORDER = PHYSICAL_MOTOR_ORDER[:5]


class _Cv2Runtime(Protocol):
    def resize(
        self,
        source: UInt8Image,
        size: tuple[int, int],
        *,
        interpolation: int,
    ) -> UInt8Image: ...


def physical_crop_to_checkpoint_image(frame_bgr: UInt8Image) -> UInt8Image:
    """Convert the audited 400x400 BGR crop to checkpoint RGB 96x96."""
    if frame_bgr.shape != (400, 400, 3) or frame_bgr.dtype != np.uint8:
        raise ValueError("physical frame must be uint8[400,400,3] BGR")
    rgb = np.ascontiguousarray(np.rot90(frame_bgr[:, :, ::-1], k=1))
    cv2_runtime = cast("_Cv2Runtime", cast("object", cv2))
    resized = cv2_runtime.resize(rgb, (96, 96), interpolation=3)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def validate_shadow_agent_pos(agent_pos: Float32Vector) -> Float32Vector:
    """Require the exact simulator state shape and dtype for shadow inference."""
    if agent_pos.shape != (5,) or agent_pos.dtype != np.float32:
        raise ValueError("shadow agent_pos must be exact float32[5]")
    if not bool(np.isfinite(agent_pos).all()):
        raise ValueError("shadow agent_pos must contain finite values")
    return agent_pos


def physical_degrees_to_shadow_agent_pos(
    positions: Mapping[str, object],
) -> Float32Vector:
    """Map calibrated physical degrees directly into provisional simulator radians."""
    if set(positions) != set(PHYSICAL_MOTOR_ORDER):
        raise ValueError(
            "physical positions must contain exactly the six calibrated motors"
        )
    degrees: list[float] = []
    for motor in SIMULATOR_JOINT_ORDER:
        value = positions[motor]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{motor} must be a finite numeric degree value")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{motor} must be a finite numeric degree value")
        degrees.append(number)
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float32))
    return validate_shadow_agent_pos(np.asarray(radians, dtype=np.float32))


def validate_shadow_action(action: NDArray[np.generic]) -> Float32Vector:
    """Validate one decoded policy action without clipping or reinterpretation."""
    if action.shape != (2,) or action.dtype != np.float32:
        raise ValueError("shadow action must be exact float32[2]")
    typed = cast("Float32Vector", cast("object", action))
    if not bool(np.isfinite(typed).all()):
        raise ValueError("shadow action must contain finite values")
    if bool(np.any(typed < -1.0)) or bool(np.any(typed > 1.0)):
        raise ValueError("shadow action exceeds [-1,1]; clipping is forbidden")
    return typed
