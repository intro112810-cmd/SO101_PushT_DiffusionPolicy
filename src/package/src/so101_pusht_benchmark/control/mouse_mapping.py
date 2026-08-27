"""Pixel to task mapping for mouse control."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class MeasuredEE:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class RequestedTarget:
    x: float
    y: float
    outside: bool


class MappingError(ValueError):
    pass


def pixels_to_task_xy(
    px: float,
    py: float,
    pane_width: float,
    pane_height: float,
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
) -> RequestedTarget:
    if not (
        math.isfinite(px)
        and math.isfinite(py)
        and math.isfinite(pane_width)
        and math.isfinite(pane_height)
    ):
        raise MappingError("Inputs must be finite")
    if pane_width <= 0 or pane_height <= 0:
        raise MappingError("Pane dimensions must be positive")

    # Task bounds dimensions
    task_w = bounds_x[1] - bounds_x[0]
    task_h = bounds_y[1] - bounds_y[0]
    if task_w <= 0 or task_h <= 0:
        raise MappingError("Task bounds must be strictly positive")

    # Determine letterbox scale and offsets to preserve aspect ratio
    scale = min(pane_width / task_w, pane_height / task_h)

    active_w = task_w * scale
    active_h = task_h * scale

    offset_x = (pane_width - active_w) / 2.0
    offset_y = (pane_height - active_h) / 2.0

    # Map pixel to active area [0, 1]
    norm_x = (px - offset_x) / active_w
    norm_y = (py - offset_y) / active_h

    # Check if outside letterbox or [0, 1] range
    outside = not (0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0)

    # Clamp normalized coordinates to [0, 1] before scaling to task space
    norm_x = max(0.0, min(1.0, norm_x))
    norm_y = max(0.0, min(1.0, norm_y))

    # Map to task space. Pixel X goes left-to-right; pixel Y is top-to-bottom
    # in the UI, so py=0 maps to bounds_y[1] (top) and py=active_h to
    # bounds_y[0] (bottom).

    task_x = bounds_x[0] + norm_x * task_w
    task_y = bounds_y[1] - norm_y * task_h  # py=0 -> bounds_y[1]

    return RequestedTarget(task_x, task_y, outside)


def task_xy_to_pixels(
    task_x: float,
    task_y: float,
    pane_width: float,
    pane_height: float,
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
) -> tuple[float, float]:
    if not (
        math.isfinite(task_x)
        and math.isfinite(task_y)
        and math.isfinite(pane_width)
        and math.isfinite(pane_height)
    ):
        raise MappingError("Inputs must be finite")
    if pane_width <= 0 or pane_height <= 0:
        raise MappingError("Pane dimensions must be positive")

    task_w = bounds_x[1] - bounds_x[0]
    task_h = bounds_y[1] - bounds_y[0]
    if task_w <= 0 or task_h <= 0:
        raise MappingError("Task bounds must be strictly positive")

    scale = min(pane_width / task_w, pane_height / task_h)
    active_w = task_w * scale
    active_h = task_h * scale
    offset_x = (pane_width - active_w) / 2.0
    offset_y = (pane_height - active_h) / 2.0

    norm_x = (task_x - bounds_x[0]) / task_w
    norm_y = (bounds_y[1] - task_y) / task_h

    px = offset_x + norm_x * active_w
    py = offset_y + norm_y * active_h

    return (px, py)
