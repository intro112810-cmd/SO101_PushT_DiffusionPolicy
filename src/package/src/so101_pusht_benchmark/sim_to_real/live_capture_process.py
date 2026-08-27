"""Linux fork-process runtime for containing blocking read-only provider calls."""

from __future__ import annotations

from multiprocessing import Process
from multiprocessing.process import BaseProcess
from pathlib import Path
import pickle
import selectors
import shutil
import socket
import tempfile
import traceback

from .live_capture_child_failure import read_child_failure, record_child_failure
from .live_capture_protocol import (
    ArmSample,
    CameraReady,
    CameraWorkerSpec,
    JointReady,
    ProviderArmed,
    ProviderCallStarted,
    ProviderClosed,
    ProviderCommand,
    ProviderEvent,
    ProviderFailed,
    ProviderProcess,
    ProviderRole,
    ProviderRuntimeReady,
    ReleaseSample,
    StartProvider,
    StopProvider,
    WorkerCompleted,
    WorkerSpec,
)
from .live_capture_workers import run_camera_worker, run_joint_worker

__all__ = ("MultiprocessingProviderRuntime",)
_HEADER_BYTES = 8


class ProcessProtocolError(RuntimeError):
    """A trusted provider child sent a malformed internal message."""


def _send(socket_value: socket.socket, value: ProviderCommand | ProviderEvent) -> None:
    payload = pickle.dumps(value, protocol=5)
    socket_value.sendall(len(payload).to_bytes(_HEADER_BYTES, "big") + payload)


def _receive_bytes(socket_value: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = socket_value.recv(size - len(chunks))
        if not chunk:
            raise EOFError
        chunks.extend(chunk)
    return bytes(chunks)


def _receive_value(socket_value: socket.socket) -> ProviderCommand | ProviderEvent:
    size = int.from_bytes(_receive_bytes(socket_value, _HEADER_BYTES), "big")
    if size <= 0:
        raise ProcessProtocolError("provider message has an invalid size")
    value = pickle.loads(_receive_bytes(socket_value, size))  # noqa: S301
    if isinstance(
        value,
        ArmSample
        | ReleaseSample
        | StartProvider
        | StopProvider
        | ProviderRuntimeReady
        | CameraReady
        | JointReady
        | ProviderArmed
        | ProviderCallStarted
        | WorkerCompleted
        | ProviderFailed
        | ProviderClosed,
    ):
        return value
    raise ProcessProtocolError("provider message variant is invalid")


def _receive_command(socket_value: socket.socket) -> ProviderCommand:
    value = _receive_value(socket_value)
    if isinstance(value, ArmSample | ReleaseSample | StartProvider | StopProvider):
        return value
    raise ProcessProtocolError("provider command variant is invalid")


def _receive_event(socket_value: socket.socket) -> ProviderEvent:
    value = _receive_value(socket_value)
    if isinstance(
        value,
        ProviderRuntimeReady
        | CameraReady
        | JointReady
        | ProviderArmed
        | ProviderCallStarted
        | WorkerCompleted
        | ProviderFailed
        | ProviderClosed,
    ):
        return value
    raise ProcessProtocolError("provider event variant is invalid")


def _emit_event(socket_value: socket.socket, failure_path: Path, event: ProviderEvent) -> None:
    if isinstance(event, ProviderFailed):
        record_child_failure(failure_path, event)
    _send(socket_value, event)


def _run_worker(spec: WorkerSpec, socket_value: socket.socket, failure_path: Path) -> None:
    role = ProviderRole.CAMERA if isinstance(spec, CameraWorkerSpec) else ProviderRole.JOINT
    try:
        if isinstance(spec, CameraWorkerSpec):
            run_camera_worker(
                spec,
                lambda: _receive_command(socket_value),
                lambda event: _emit_event(socket_value, failure_path, event),
            )
        else:
            run_joint_worker(
                spec,
                lambda: _receive_command(socket_value),
                lambda event: _emit_event(socket_value, failure_path, event),
            )
    except (AssertionError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        failure = ProviderFailed(
            role,
            "child_process",
            None,
            spec.clock(),
            type(exc).__name__,
            str(exc),
            traceback.format_exc(),
        )
        record_child_failure(failure_path, failure)
        try:
            _send(socket_value, failure)
        except OSError as send_error:
            del send_error
    finally:
        socket_value.close()


class _MultiprocessingProviderProcess:
    def __init__(
        self,
        role: ProviderRole,
        process: BaseProcess,
        socket_value: socket.socket,
        failure_path: Path,
        failure_root: Path,
    ) -> None:
        self.role = role
        self._process = process
        self._socket = socket_value
        self._failure_path = failure_path
        self._failure_root = failure_root

    @property
    def socket(self) -> socket.socket:
        return self._socket

    def start(self) -> None:
        self._process.start()

    def send(self, command: ProviderCommand) -> None:
        _send(self._socket, command)

    def receive(self) -> ProviderEvent:
        return _receive_event(self._socket)

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()

    def join(self, timeout: float) -> bool:
        self._process.join(timeout)
        return not self._process.is_alive()

    def exit_code(self) -> int | None:
        return self._process.exitcode

    def child_failure(self) -> ProviderFailed | None:
        return read_child_failure(self._failure_path, self.role)

    def close(self) -> None:
        self._socket.close()
        self._process.close()
        shutil.rmtree(self._failure_root)


class MultiprocessingProviderRuntime:
    """Spawn non-daemon fork workers and await their sockets without polling."""

    def spawn(self, spec: WorkerSpec) -> ProviderProcess:
        parent_socket, child_socket = socket.socketpair()
        role = ProviderRole.CAMERA if isinstance(spec, CameraWorkerSpec) else ProviderRole.JOINT
        failure_root = Path(tempfile.mkdtemp(prefix="so101-live-child-"))
        failure_path = failure_root / "failure.json"
        process = Process(
            target=_run_worker,
            args=(spec, child_socket, failure_path),
            daemon=False,
        )
        return _MultiprocessingProviderProcess(
            role,
            process,
            parent_socket,
            failure_path,
            failure_root,
        )

    def wait(
        self,
        processes: tuple[ProviderProcess, ...],
        timeout: float,
    ) -> tuple[ProviderProcess, ...]:
        concrete: list[_MultiprocessingProviderProcess] = []
        with selectors.DefaultSelector() as selector:
            for process in processes:
                if not isinstance(process, _MultiprocessingProviderProcess):
                    raise ProcessProtocolError("process belongs to another runtime")
                concrete.append(process)
                selector.register(process.socket, selectors.EVENT_READ, process)
            selected = selector.select(max(0.0, timeout))
        ready_sockets = {key.fileobj for key, _mask in selected}
        return tuple(process for process in concrete if process.socket in ready_sockets)
