"""Fresh camera-frame source used only after a production write attempt."""

from __future__ import annotations
import hashlib
from importlib import import_module
from pathlib import Path
import time
from typing import Protocol, cast
from .rollout_codes import RolloutCode, RolloutViolation


class Capture(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, object]: ...
    def release(self) -> None: ...


class EncodedBuffer(Protocol):
    def tobytes(self) -> bytes: ...


class Cv2(Protocol):
    def VideoCapture(self, path: str) -> Capture: ...
    def imencode(self, extension: str, frame: object) -> tuple[bool, EncodedBuffer]: ...


class OpenCvPostFrameSource:
    def __init__(self, device: Path, output: Path) -> None:
        self._device = device
        self._output = output

    def __call__(self) -> tuple[float, str]:
        cv2 = cast("Cv2", import_module("cv2"))
        capture = cv2.VideoCapture(str(self._device))
        try:
            if not capture.isOpened():
                raise RolloutViolation(RolloutCode.R_POST_STATE_MISSING, "post camera unavailable")
            ok, frame = capture.read()
            if not ok:
                raise RolloutViolation(RolloutCode.R_POST_STATE_MISSING, "post frame unavailable")
            encoded_ok, encoded = cv2.imencode(".png", frame)
            if not encoded_ok:
                raise RolloutViolation(RolloutCode.F_POST_STATE_INVALID, "post frame encode")
            content = encoded.tobytes()
            self._output.write_bytes(content)
            return time.monotonic(), hashlib.sha256(content).hexdigest()
        finally:
            capture.release()
