from __future__ import annotations

import ast
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace
from typing import Protocol, cast

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).parents[1]
DEFAULT_UPSTREAM_SOURCE = (
    PACKAGE_ROOT.parents[1] / "05_references/external_repos/pushT-so100/src/env_human_ee.py"
)


class JoystickControl(Protocol):
    def __call__(self) -> bool | int: ...


@dataclass
class FakeJoystick:
    axes: dict[int, float] = field(default_factory=dict)
    buttons: dict[int, int] = field(default_factory=dict)
    axis_reads: list[int] = field(default_factory=list)
    button_reads: list[int] = field(default_factory=list)

    def get_axis(self, index: int) -> float:
        self.axis_reads.append(index)
        return self.axes.get(index, 0.0)

    def get_button(self, index: int) -> int:
        self.button_reads.append(index)
        return self.buttons.get(index, 0)


@dataclass(frozen=True)
class FakeRotation:
    quat_xyzw: tuple[float, float, float, float]

    @classmethod
    def from_quat(cls, values: Sequence[float]) -> FakeRotation:
        assert len(values) == 4
        return cls((float(values[0]), float(values[1]), float(values[2]), float(values[3])))

    @classmethod
    def from_euler(cls, axis: str, angle: float) -> FakeRotation:
        assert axis == "y"
        half = angle / 2.0
        return cls((0.0, math.sin(half), 0.0, math.cos(half)))

    def __mul__(self, other: FakeRotation) -> FakeRotation:
        """Compose two fake rotations using Hamilton multiplication."""
        x1, y1, z1, w1 = self.quat_xyzw
        x2, y2, z2, w2 = other.quat_xyzw
        return FakeRotation(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            )
        )

    def as_quat(self) -> tuple[float, float, float, float]:
        return self.quat_xyzw


@dataclass
class FakeClock:
    now: float

    def time(self) -> float:
        return self.now


@dataclass
class Harness:
    control: JoystickControl
    joystick: FakeJoystick
    data: SimpleNamespace
    clock: FakeClock
    namespace: dict[str, object]
    calls: list[str]
    pump_calls: list[str]


def _source_path() -> Path:
    override = os.environ.get("F710_SOURCE_OVERRIDE")
    return Path(override) if override else DEFAULT_UPSTREAM_SOURCE


def _joystick_function(source_path: Path) -> ast.FunctionDef:
    source = source_path.read_text(encoding="utf-8")
    if os.environ.get("F710_MUTATION_AXIS_3_TO_2") == "1":
        source = source.replace("get_axis(3)", "get_axis(2)", 1)
    module = ast.parse(source, filename=str(source_path))
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "joystick_control"
    ]
    assert len(functions) == 1
    return functions[0]


def _harness(
    *,
    axes: dict[int, float] | None = None,
    buttons: dict[int, int] | None = None,
    now: float = 10.0,
    timestep: float = 0.02,
) -> Harness:
    joystick = FakeJoystick(axes=axes or {}, buttons=buttons or {})
    data = SimpleNamespace(
        mocap_pos=np.zeros((1, 3), dtype=np.float64),
        mocap_quat=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    clock = FakeClock(now)
    calls: list[str] = []
    pump_calls: list[str] = []
    namespace: dict[str, object] = {
        "buttonCooldown": 0.0,
        "joystick": joystick,
        "pygame": SimpleNamespace(event=SimpleNamespace(pump=lambda: pump_calls.append("pump"))),
        "DEADZONE": 0.1,
        "MOVE_SPEED": 0.05,
        "ROT_SPEED": 1.0,
        "COOLDOWN_SEC": 0.3,
        "model": SimpleNamespace(opt=SimpleNamespace(timestep=timestep)),
        "data": data,
        "mocap_id": 0,
        "np": np,
        "R": FakeRotation,
        "time": clock,
        "reset_env": lambda: calls.append("reset"),
        "record_toggle": lambda: calls.append("record_toggle"),
    }
    function = _joystick_function(_source_path())
    compiled = compile(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        filename=str(_source_path()),
        mode="exec",
    )
    function_codes = [
        value
        for value in compiled.co_consts
        if isinstance(value, CodeType) and value.co_name == "joystick_control"
    ]
    assert len(function_codes) == 1
    control = FunctionType(function_codes[0], namespace)
    return Harness(
        control=cast("JoystickControl", cast("object", control)),
        joystick=joystick,
        data=data,
        clock=clock,
        namespace=namespace,
        calls=calls,
        pump_calls=pump_calls,
    )


def test_f710_axes_xy_and_buttons_z_preserve_deadzone_and_speed_scaling() -> None:
    harness = _harness(axes={0: 0.5, 1: -0.25}, buttons={4: 1, 0: 0})

    assert harness.control() == 0

    np.testing.assert_allclose(harness.data.mocap_pos[0], [0.0005, 0.00025, 0.001])
    assert harness.joystick.axis_reads[:2] == [0, 1]
    assert harness.joystick.button_reads[:2] == [4, 0]
    assert harness.pump_calls == ["pump"]

    at_deadzone = _harness(axes={0: 0.1, 1: -0.1}, buttons={4: 1, 0: 1})
    assert at_deadzone.control() == 0
    np.testing.assert_array_equal(at_deadzone.data.mocap_pos[0], np.zeros(3))


def test_f710_axis_3_rotation_preserves_deadzone_and_rotation_speed_scaling() -> None:
    harness = _harness(axes={2: 0.0, 3: 0.5})

    assert harness.control() == 0

    expected_wxyz = [math.cos(-0.005), 0.0, math.sin(-0.005), 0.0]
    np.testing.assert_allclose(harness.data.mocap_quat[0], expected_wxyz)
    assert harness.joystick.axis_reads == [0, 1, 3]
    assert 2 not in harness.joystick.axis_reads

    at_deadzone = _harness(axes={3: -0.1})
    assert at_deadzone.control() == 0
    np.testing.assert_array_equal(at_deadzone.data.mocap_quat[0], [1.0, 0.0, 0.0, 0.0])


def test_f710_reset_record_toggle_exit_and_shared_strict_debounce() -> None:
    reset = _harness(buttons={3: 1})
    assert reset.control() == 0
    assert reset.calls == ["reset"]

    reset.clock.now = 10.2
    assert reset.control() == 0
    assert reset.calls == ["reset"]

    reset.clock.now = 10.31
    assert reset.control() == 0
    assert reset.calls == ["reset", "reset"]

    record = _harness(buttons={1: 1})
    assert record.control() == 0
    assert record.calls == ["record_toggle"]

    shared = _harness(buttons={3: 1, 1: 1})
    assert shared.control() == 0
    assert shared.calls == ["reset"]

    exiting = _harness(buttons={7: 1}, now=0.0)
    assert exiting.control() == 1
    assert exiting.calls == []
    assert exiting.joystick.button_reads[-1] == 7


def test_no_joystick_returns_without_pumping_or_mutating() -> None:
    harness = _harness()
    harness.namespace["joystick"] = None

    assert harness.control() is False

    assert harness.pump_calls == []
    np.testing.assert_array_equal(harness.data.mocap_pos[0], np.zeros(3))


@pytest.mark.parametrize(
    ("start", "axes", "expected_xy"),
    [
        ((0.9999, 0.0, 0.0), {0: 1.0}, (1.0, 0.0)),
        ((-0.9999, 0.0, 0.0), {0: -1.0}, (-1.0, 0.0)),
        ((0.0, 0.9999, 0.0), {1: -1.0}, (0.0, 1.0)),
        ((0.0, -0.9999, 0.0), {1: 1.0}, (0.0, -1.0)),
    ],
)
def test_f710_xy_actions_remain_inside_persisted_contract(
    start: tuple[float, float, float],
    axes: dict[int, float],
    expected_xy: tuple[float, float],
) -> None:
    harness = _harness(axes=axes)
    harness.data.mocap_pos[0] = start

    assert harness.control() == 0

    np.testing.assert_allclose(harness.data.mocap_pos[0, :2], expected_xy)
