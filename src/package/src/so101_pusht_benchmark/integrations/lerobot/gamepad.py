"""Read-only adapter for LeRobot 0.4.4 public ``GamepadTeleop``."""

from __future__ import annotations
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast


@dataclass(frozen=True, slots=True)
class GamepadSample:
    axes: tuple[float, float, float]
    deadman: bool
    success: bool = False
    stop: bool = False
    rerecord: bool = False
    connected: bool = True
    fresh: bool = True

    def __post_init__(self) -> None:
        """Reject non-XYZ gamepad samples at the adapter boundary."""
        if type(self.axes) is not tuple or len(self.axes) != 3:
            raise ValueError("gamepad axes must be an XYZ 3-tuple")


class EventSource(Protocol):
    def poll(self) -> GamepadSample: ...
    def close(self) -> None: ...


class _GamepadDevice(Protocol):
    running: bool

    def update(self) -> None: ...
    def get_deltas(self) -> tuple[float, float, float]: ...
    def should_intervene(self) -> bool: ...
    def get_episode_end_status(self) -> str | None: ...


class _GamepadTeleop(Protocol):
    gamepad: _GamepadDevice | None
    is_connected: bool

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_action(self) -> dict[str, float]: ...
    def get_teleop_events(self) -> dict[str, bool]: ...


class _GamepadConfigConstructor(Protocol):
    def __call__(self, use_gripper: bool = ...) -> object: ...


class _GamepadTeleopConstructor(Protocol):
    def __call__(self, config: object) -> object: ...


class _GamepadModule(Protocol):
    GamepadTeleop: _GamepadTeleopConstructor
    GamepadTeleopConfig: _GamepadConfigConstructor


class PublicGamepadSource:
    """Uses only LeRobot's public GamepadTeleop configuration and methods."""

    def __init__(self) -> None:
        try:
            module = cast(_GamepadModule, import_module("lerobot.teleoperators.gamepad"))
        except ImportError as exc:
            raise RuntimeError("LeRobot 0.4.4 is required; no fallback is permitted") from exc
        config = module.GamepadTeleopConfig(use_gripper=False)
        self._teleop = cast(_GamepadTeleop, module.GamepadTeleop(config))
        self._teleop.connect()
        gamepad = self._teleop.gamepad
        if gamepad is None or not gamepad.running:
            self._teleop.disconnect()
            raise RuntimeError("no physical gamepad detected")
        self._neutral = True

    def poll(self) -> GamepadSample:
        gamepad = self._teleop.gamepad
        if gamepad is None or not gamepad.running:
            return GamepadSample((0.0, 0.0, 0.0), False, connected=False)
        gamepad.update()
        delta_x, delta_y, delta_z = gamepad.get_deltas()
        axes = (float(delta_x), float(delta_y), float(delta_z))
        status = gamepad.get_episode_end_status()
        if self._neutral:
            self._neutral = max(abs(axis) for axis in axes) > 0.0
            return GamepadSample((0.0, 0.0, 0.0), False)
        return GamepadSample(
            axes,
            gamepad.should_intervene(),
            status == "success",
            status == "failure",
            status == "rerecord_episode",
        )

    def close(self) -> None:
        self._teleop.disconnect()
