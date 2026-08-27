"""Non-actuating live path and calibration checks for the real SO-101 setup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

import json
from pathlib import Path

EXPECTED_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class _Ported(Protocol):
    @property
    def port(self) -> Path: ...

    @property
    def calibration_file(self) -> Path: ...


class _Camera(Protocol):
    @property
    def device(self) -> Path: ...

    @property
    def latest_frame(self) -> Path: ...


class _LiveProfile(Protocol):
    @property
    def follower(self) -> _Ported: ...

    @property
    def leader(self) -> _Ported: ...

    @property
    def camera(self) -> _Camera: ...


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, object], value)


def _integer(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _valid_calibration(path: Path) -> bool:
    if not path.is_file():
        return False
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), "calibration")
    if tuple(raw) != EXPECTED_MOTORS:
        return False
    ids = [_integer(_mapping(raw[name], f"calibration.{name}"), "id") for name in EXPECTED_MOTORS]
    return ids == list(range(1, 7))


def live_checks(profile: _LiveProfile) -> dict[str, bool]:
    """Check paths and calibration contents without opening buses or camera."""
    return {
        "follower_port": profile.follower.port.exists(),
        "leader_port": profile.leader.port.exists(),
        "follower_calibration": _valid_calibration(profile.follower.calibration_file),
        "leader_calibration": _valid_calibration(profile.leader.calibration_file),
        "camera_device": profile.camera.device.exists(),
        "camera_latest_frame": profile.camera.latest_frame.is_file(),
    }


def _process_holds_device(process: Path, target: Path) -> bool:
    try:
        return any(descriptor.resolve() == target for descriptor in (process / "fd").iterdir())
    except (FileNotFoundError, PermissionError):
        return False


def device_holders(device: Path) -> tuple[int, ...]:
    """Return processes with an open descriptor to a device, without signaling."""
    target = device.resolve()
    holders = [
        int(process.name)
        for process in Path("/proc").glob("[0-9]*")
        if _process_holds_device(process, target)
    ]
    return tuple(sorted(holders))
