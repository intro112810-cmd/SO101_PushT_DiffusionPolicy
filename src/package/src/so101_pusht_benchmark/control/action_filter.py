"""Fail-closed Cartesian action shaping, independent of simulation packages."""

from __future__ import annotations
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..task.spec import APPROACH_HEIGHT_M, CONTACT_HEIGHT_M, EE_Z_BOUNDS_M


@dataclass(frozen=True, slots=True)
class FilterResult:
    requested: tuple[float, float, float]
    applied: tuple[float, float, float]
    clipped: bool


class ActionFilter:
    """Apply workspace clamp then bounded slew from the last applied target."""

    def __init__(
        self,
        max_step: float,
        initial: tuple[float, float, float] = (0.28, 0.0, APPROACH_HEIGHT_M),
    ) -> None:
        self.max_step = max_step
        self.last: NDArray[np.float64] = np.asarray(initial, dtype=np.float64)

    def clear(self) -> None:
        self.last.fill(0.0)

    def apply(self, value: object) -> FilterResult:
        if type(value) is not np.ndarray:
            raise ValueError("action must be finite float32[3]")
        action = cast("NDArray[np.float32]", value)
        if action.shape != (3,) or action.dtype != np.float32:
            raise ValueError("action must be finite float32[3]")
        requested: NDArray[np.float64] = action.astype(np.float64, copy=True)
        if not bool(np.isfinite(requested).all()):
            raise ValueError("action must be finite")
        bounded = requested.copy()
        for boundary in (CONTACT_HEIGHT_M, APPROACH_HEIGHT_M):
            if action[2] == np.float32(boundary):
                bounded[2] = boundary
        bounded = np.clip(
            bounded,
            (0.18, -0.16, EE_Z_BOUNDS_M[0]),
            (0.38, 0.16, EE_Z_BOUNDS_M[1]),
        )
        delta: NDArray[np.float64] = np.clip(bounded - self.last, -self.max_step, self.max_step)
        bounded = self.last + delta
        result = FilterResult(
            (float(requested[0]), float(requested[1]), float(requested[2])),
            (float(bounded[0]), float(bounded[1]), float(bounded[2])),
            not np.array_equal(requested, bounded),
        )
        self.last = bounded.copy()
        return result
