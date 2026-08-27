"""Exact read-only lifecycle contracts for live camera and follower adapters."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from so101_pusht_benchmark.sim_to_real.live_capture_adapters import (
    DirectBusJointReader,
    ReadOnlyOpenCvCamera,
)
from so101_pusht_benchmark.sim_to_real.live_capture_types import (
    AdapterIdentity,
    LiveCaptureConfiguration,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/so101_pusht_benchmark/sim_to_real/live_capture_adapters.py"
SHA_PROVIDER = "1" * 64
SHA_CAMERA = "2" * 64
SHA_FOLLOWER = "3" * 64
SHA_CALIBRATION = "4" * 64


class SequenceClock:
    def __init__(
        self,
        values: list[float],
        log: list[tuple[str, object]] | None = None,
    ) -> None:
        self._values = iter(values)
        self._log = log

    def __call__(self) -> float:
        value = next(self._values)
        if self._log is not None:
            self._log.append(("clock", value))
        return value


class FakeFrame:
    def __init__(self, log: list[tuple[str, object]]) -> None:
        self._log = log

    def tobytes(self) -> bytes:
        self._log.append(("camera.serialize", None))
        return b"raw-camera-frame"


class FakeCapture:
    def __init__(
        self,
        log: list[tuple[str, object]],
        *,
        opened: bool = True,
        invalid_profile: bool = False,
    ) -> None:
        self.log = log
        self.opened = opened
        self.invalid_profile = invalid_profile

    def isOpened(self) -> bool:
        self.log.append(("camera.isOpened", None))
        return self.opened

    def get(self, prop: int) -> float:
        self.log.append(("camera.get", prop))
        if self.invalid_profile:
            return float("nan")
        return {1: 640.0, 2: 480.0, 3: 30.0}[prop]

    def read(self) -> tuple[bool, FakeFrame]:
        self.log.append(("camera.read", None))
        return True, FakeFrame(self.log)

    def release(self) -> None:
        self.log.append(("camera.release", None))

    def set(self, _prop: int, _value: float) -> None:
        raise AssertionError("camera property setters are forbidden")


class FakeBus:
    def __init__(self, log: list[tuple[str, object]], *, fail: bool = False) -> None:
        self.log = log
        self.fail = fail

    def connect(self) -> None:
        self.log.append(("bus.connect", None))

    def sync_read(self, register: str) -> dict[str, float]:
        self.log.append(("bus.sync_read", register))
        if self.fail:
            raise RuntimeError("read failed")
        return {
            "shoulder_pan": 0.0,
            "shoulder_lift": 1.0,
            "elbow_flex": 2.0,
            "wrist_flex": 3.0,
            "wrist_roll": 4.0,
            "gripper": 5.0,
        }

    def disconnect(self, *, disable_torque: bool) -> None:
        self.log.append(("bus.disconnect", disable_torque))

    def sync_write(self, _register: str, _payload: object) -> None:
        raise AssertionError("register writes are forbidden")


class FakeRobot:
    def __init__(self, bus: FakeBus) -> None:
        self.bus = bus

    def connect(self) -> None:
        raise AssertionError("SOFollower.connect is forbidden")

    def configure(self) -> None:
        raise AssertionError("SOFollower.configure is forbidden")

    def calibrate(self) -> None:
        raise AssertionError("SOFollower.calibrate is forbidden")

    def send_action(self) -> None:
        raise AssertionError("SOFollower.send_action is forbidden")


def _configuration(tmp_path: Path) -> LiveCaptureConfiguration:
    return LiveCaptureConfiguration(
        tmp_path / "profile.yaml",
        tmp_path / "camera",
        tmp_path / "follower",
        tmp_path / "calibration.json",
        640,
        480,
        30.0,
    )


def test_camera_opens_existing_capture_gets_properties_and_never_sets(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    capture = FakeCapture(log)

    def factory(path: str) -> FakeCapture:
        log.append(("VideoCapture", path))
        return capture

    camera = ReadOnlyOpenCvCamera(
        _configuration(tmp_path),
        AdapterIdentity(SHA_PROVIDER, SHA_CAMERA, None),
        capture_factory=factory,
        property_ids=(1, 2, 3),
        clock=SequenceClock([1000.0, 1000.002], log),
    )
    observed = camera.open()
    frame = camera.next_frame()
    camera.close()

    assert (observed.width, observed.height, observed.fps) == (640, 480, 30.0)
    assert frame.frame_bytes == b"raw-camera-frame"
    assert (frame.started_at, frame.completed_at) == (1000.0, 1000.002)
    assert log == [
        ("VideoCapture", str(tmp_path / "camera")),
        ("camera.isOpened", None),
        ("camera.get", 1),
        ("camera.get", 2),
        ("camera.get", 3),
        ("clock", 1000.0),
        ("camera.read", None),
        ("camera.serialize", None),
        ("clock", 1000.002),
        ("camera.release", None),
    ]


def test_camera_open_failure_releases_capture(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    capture = FakeCapture(log, opened=False)
    camera = ReadOnlyOpenCvCamera(
        _configuration(tmp_path),
        AdapterIdentity(SHA_PROVIDER, SHA_CAMERA, None),
        capture_factory=lambda _path: capture,
        property_ids=(1, 2, 3),
        clock=SequenceClock([]),
    )

    with pytest.raises(RuntimeError, match="camera open failed"):
        camera.open()
    assert log == [("camera.isOpened", None), ("camera.release", None)]


def test_camera_profile_failure_releases_capture(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    capture = FakeCapture(log, invalid_profile=True)
    camera = ReadOnlyOpenCvCamera(
        _configuration(tmp_path),
        AdapterIdentity(SHA_PROVIDER, SHA_CAMERA, None),
        capture_factory=lambda _path: capture,
        property_ids=(1, 2, 3),
        clock=SequenceClock([]),
    )

    with pytest.raises(ValueError, match="observed camera profile"):
        camera.open()
    assert log[-1] == ("camera.release", None)


def test_joint_adapter_connects_once_for_exactly_two_reads_and_disconnects_once() -> None:
    log: list[tuple[str, object]] = []
    reader = DirectBusJointReader(
        FakeRobot(FakeBus(log)),
        AdapterIdentity(SHA_PROVIDER, SHA_FOLLOWER, SHA_CALIBRATION),
        clock=SequenceClock([1000.003, 1000.005, 1000.023, 1000.025]),
    )

    reader.open()
    first = reader.next_state()
    second = reader.next_state()
    reader.close()

    assert first.read_id == "joint-000"
    assert second.read_id == "joint-001"
    assert first.body_degrees == (0.0, 1.0, 2.0, 3.0, 4.0)
    assert log == [
        ("bus.connect", None),
        ("bus.sync_read", "Present_Position"),
        ("bus.sync_read", "Present_Position"),
        ("bus.disconnect", False),
    ]


def test_joint_read_failure_disconnects_without_retry() -> None:
    log: list[tuple[str, object]] = []
    reader = DirectBusJointReader(
        FakeRobot(FakeBus(log, fail=True)),
        AdapterIdentity(SHA_PROVIDER, SHA_FOLLOWER, SHA_CALIBRATION),
        clock=SequenceClock([1000.003]),
    )

    reader.open()
    with pytest.raises(RuntimeError, match="read failed"):
        reader.next_state()
    reader.close()
    assert log == [
        ("bus.connect", None),
        ("bus.sync_read", "Present_Position"),
        ("bus.disconnect", False),
    ]


def test_adapter_source_has_no_setters_writes_or_robot_lifecycle_calls() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
    assert called_attributes.isdisjoint(
        {"set", "sync_write", "configure", "calibrate", "send_action"}
    )
    source = SOURCE.read_text(encoding="utf-8")
    assert "Goal_Position" not in source
    assert "disable_torque=True" not in source
