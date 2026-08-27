"""Concrete non-configuring camera and direct-bus joint read adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from typing import Protocol

from .joint_mapping import JOINT_ORDER
from .live_capture_types import (
    AdapterIdentity,
    CameraObservation,
    LiveCaptureConfiguration,
    TimedCameraRead,
    TimedJointRead,
)
from .rollout_codes import RolloutCode, RolloutViolation
from .sample_capture import Clock

__all__ = (
    "CaptureFactory",
    "DirectBusJointReader",
    "DirectReadRobot",
    "ReadOnlyOpenCvCamera",
)
_EXPECTED_MOTORS = frozenset((*JOINT_ORDER, "gripper"))


class _Frame(Protocol):
    def tobytes(self) -> bytes: ...


class _Capture(Protocol):
    def isOpened(self) -> bool: ...

    def get(self, property_id: int, /) -> float: ...

    def read(self) -> tuple[bool, _Frame]: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[str], _Capture]


class _DirectReadBus(Protocol):
    def connect(self) -> None: ...

    def sync_read(self, register: str) -> Mapping[str, float]: ...

    def disconnect(self, *, disable_torque: bool) -> None: ...


class DirectReadRobot(Protocol):
    @property
    def bus(self) -> _DirectReadBus: ...


class ReadOnlyOpenCvCamera:
    """Open an existing capture, observe properties, and expose only frame reads."""

    def __init__(
        self,
        configuration: LiveCaptureConfiguration,
        identity: AdapterIdentity,
        *,
        capture_factory: CaptureFactory,
        property_ids: tuple[int, int, int],
        clock: Clock,
    ) -> None:
        self.identity = identity
        self._configuration = configuration
        self._capture_factory = capture_factory
        self._property_ids = property_ids
        self._clock = clock
        self._capture: _Capture | None = None
        self._read_index = 0

    def open(self) -> CameraObservation:
        """Open without passing or changing width, height, FPS, or properties."""
        if self._capture is not None:
            raise RuntimeError("camera is already open")
        capture = self._capture_factory(str(self._configuration.camera_device))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("camera open failed")
        self._capture = capture
        try:
            width_id, height_id, fps_id = self._property_ids
            width = capture.get(width_id)
            height = capture.get(height_id)
            fps = capture.get(fps_id)
            if not all(math.isfinite(value) for value in (width, height, fps)):
                raise RolloutViolation(RolloutCode.R_NONFINITE, "observed camera profile")
            if not width.is_integer() or not height.is_integer():
                raise RolloutViolation(
                    RolloutCode.CAMERA_UNREGISTERED, "non-integral camera profile"
                )
            return CameraObservation(int(width), int(height), fps)
        except Exception:
            self._capture = None
            capture.release()
            raise

    def next_frame(self) -> TimedCameraRead:
        """Timestamp immediately before and after one byte-preserving read."""
        capture = self._capture
        if capture is None:
            raise RuntimeError("camera is not open")
        started_at = self._clock()
        success, frame = capture.read()
        if not success:
            raise RuntimeError("camera read failed")
        frame_bytes = frame.tobytes()
        completed_at = self._clock()
        index = self._read_index
        self._read_index += 1
        return TimedCameraRead(
            f"camera-{index:03d}",
            frame_bytes,
            started_at,
            completed_at,
        )

    def close(self) -> None:
        """Release exactly the capture opened by this adapter."""
        capture = self._capture
        if capture is None:
            return
        self._capture = None
        capture.release()


class DirectBusJointReader:
    """Read Present_Position over one persistent direct-bus connection."""

    def __init__(
        self,
        robot: DirectReadRobot,
        identity: AdapterIdentity,
        *,
        clock: Clock,
    ) -> None:
        self.identity = identity
        self._robot = robot
        self._clock = clock
        self._read_index = 0
        self._open = False

    def open(self) -> None:
        """Connect the direct bus once without invoking robot lifecycle APIs."""
        if self._open:
            raise RuntimeError("joint reader is already open")
        self._robot.bus.connect()
        self._open = True

    def next_state(self) -> TimedJointRead:
        """Read and serialize one Present_Position result on the open bus."""
        if not self._open:
            raise RuntimeError("joint reader is not open")
        started_at = self._clock()
        positions = self._robot.bus.sync_read("Present_Position")
        if frozenset(positions) != _EXPECTED_MOTORS:
            raise RolloutViolation(RolloutCode.R_PROVIDER_MISMATCH, "joint provider motor set")
        body = tuple(float(positions[name]) for name in JOINT_ORDER)
        if len(body) != 5 or not all(math.isfinite(value) for value in body):
            raise RolloutViolation(RolloutCode.R_NONFINITE, "joint provider values")
        completed_at = self._clock()
        index = self._read_index
        self._read_index += 1
        return TimedJointRead(
            f"joint-{index:03d}",
            body,
            started_at,
            completed_at,
        )

    def close(self) -> None:
        """Disconnect once while preserving the robot's current torque state."""
        if not self._open:
            return
        self._open = False
        self._robot.bus.disconnect(disable_torque=False)
