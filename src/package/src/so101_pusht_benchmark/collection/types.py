"""Typed collection events and raw frame construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..data.store import FrameRecord
from ..task.spec import APPROACH_HEIGHT_M, CONTACT_HEIGHT_M

if TYPE_CHECKING:
    from numpy import generic
    from numpy.typing import NDArray


class RecorderState(str, Enum):
    DISCONNECTED = "disconnected"
    NEUTRAL_REQUIRED = "neutral_required"
    ARMED = "armed"
    STOPPED = "stopped"
    FAULT = "fault"


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    deadzone: float
    xy_meters_per_tick: float
    stale_timeout_s: float
    debounce_ticks: int
    z_axis: int
    z_meters_per_tick: float
    z_contact_m: float
    z_approach_m: float

    def __post_init__(self) -> None:
        """Keep the collection Z envelope identical to the task contract."""
        if (
            self.z_axis != 2
            or self.z_meters_per_tick <= 0
            or (
                self.z_contact_m,
                self.z_approach_m,
            )
            != (CONTACT_HEIGHT_M, APPROACH_HEIGHT_M)
        ):
            raise ValueError("collection Z envelope must match the task specification")

    @classmethod
    def load(cls, path: Path) -> CollectionConfig:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError("invalid frozen collection config")
        raw = cast("dict[str, object]", loaded)
        if raw.get("schema") != 2 or raw.get("fps") != 10:
            raise ValueError("invalid frozen collection config")
        keys = (
            "deadzone",
            "xy_meters_per_tick",
            "stale_timeout_s",
            "button_debounce_ticks",
            "z_axis",
            "z_meters_per_tick",
            "z_contact_m",
            "z_approach_m",
        )
        values = tuple(raw.get(key) for key in keys)
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError("collection config values must be numeric")
        typed = cast("tuple[float, float, float, int, int, float, float, float]", values)
        return cls(
            float(typed[0]),
            float(typed[1]),
            float(typed[2]),
            int(typed[3]),
            int(typed[4]),
            float(typed[5]),
            float(typed[6]),
            float(typed[7]),
        )


@dataclass(frozen=True, slots=True)
class MouseCollectionConfig:
    """Schema-3 mouse/keyboard collection parameters."""

    stale_timeout_s: float
    debounce_ticks: int
    contact_z_m: float
    clearance_z_m: float
    bounds_x: tuple[float, float]
    bounds_y: tuple[float, float]

    @classmethod
    def load(cls, path: Path) -> MouseCollectionConfig:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError("invalid mouse collection config")
        raw = cast("dict[str, object]", loaded)
        if raw.get("schema") != 3 or raw.get("source") != "human_mouse_keyboard":
            raise ValueError("mouse collection config must be schema 3 human_mouse_keyboard")
        z_levels_raw = raw.get("z_levels")
        bounds_raw = raw.get("task_bounds")
        if not isinstance(z_levels_raw, dict) or not isinstance(bounds_raw, dict):
            raise TypeError("mouse collection config must define z_levels and task_bounds")
        z_levels = cast("dict[str, object]", z_levels_raw)
        bounds = cast("dict[str, object]", bounds_raw)
        contact = cast("float", z_levels.get("contact_z_m"))
        clearance = cast("float", z_levels.get("clearance_z_m"))
        bx_raw = bounds.get("x")
        by_raw = bounds.get("y")
        if not isinstance(bx_raw, list) or not isinstance(by_raw, list):
            raise TypeError("mouse task bounds must be [min, max] pairs")
        bx_list = cast("list[object]", bx_raw)
        by_list = cast("list[object]", by_raw)
        if len(bx_list) != 2 or len(by_list) != 2:
            raise ValueError("mouse task bounds must be [min, max] numeric pairs")
        raw_bounds: list[object] = [
            contact,
            clearance,
            bx_list[0],
            bx_list[1],
            by_list[0],
            by_list[1],
        ]
        if not all(isinstance(value, (int, float)) for value in raw_bounds):
            raise ValueError("mouse collection config z/task bounds must be numeric")
        bx = cast("list[float]", bx_raw)
        by = cast("list[float]", by_raw)
        stale = cast("float", raw.get("stale_timeout_s", 0.35))
        debounce = cast("int", raw.get("button_debounce_ticks", 2))
        return cls(
            float(stale),
            int(debounce),
            float(contact),
            float(clearance),
            (float(bx[0]), float(bx[1])),
            (float(by[0]), float(by[1])),
        )


def fault_frame(
    index: int,
    observation: dict[str, NDArray[generic]],
    target: tuple[float, float, float],
    raw_axes: object,
    reason: str,
) -> FrameRecord:
    return FrameRecord(
        index,
        index / 10,
        observation,
        target,
        raw_axes,
        target,
        {
            "raw_axes": raw_axes,
            "raw_sample_available": raw_axes is not None,
            "requested_target": target,
            "applied_action": target,
            "fault": reason,
            "applied": False,
            "command_id": index * 2,
            "frame_id": index * 2 + 1,
            "observation_timestamp": index / 10,
            "action_timestamp": index / 10,
            "next_state_timestamp": index / 10,
        },
        observation,
        False,
    )
