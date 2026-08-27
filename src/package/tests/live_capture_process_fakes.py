"""Deterministic in-process implementation of the provider-process protocol."""

from __future__ import annotations

from collections import deque

from so101_pusht_benchmark.sim_to_real.live_capture_protocol import (
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
from so101_pusht_benchmark.sim_to_real.live_capture_types import (
    LiveCameraReader,
    LiveJointReader,
)
from so101_pusht_benchmark.sim_to_real.live_capture_validation import (
    require_adapter_identity,
    require_camera_profile,
)


class FakeProviderProcess:
    """Mutable command-driven fake whose queue behaves like one child pipe."""

    def __init__(self, spec: WorkerSpec) -> None:
        self.role = (
            ProviderRole.CAMERA if isinstance(spec, CameraWorkerSpec) else ProviderRole.JOINT
        )
        self._spec = spec
        self._events: deque[ProviderEvent] = deque()
        self._alive = False
        self._camera: LiveCameraReader | None = None
        self._joint: LiveJointReader | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_calls = 0
        self.release_count = 0
        self.provider_started = False

    @property
    def has_event(self) -> bool:
        return bool(self._events)

    def start(self) -> None:
        self._alive = True
        started = self._spec.clock()
        try:
            dependency = self._spec.runtime_preflight()
            self._events.append(
                ProviderRuntimeReady(self.role, started, self._spec.clock(), dependency)
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._events.append(
                ProviderFailed(
                    self.role,
                    "runtime_preflight",
                    None,
                    self._spec.clock(),
                    type(exc).__name__,
                    str(exc),
                    "fixture traceback",
                )
            )

    def _start_provider(self) -> None:
        self.provider_started = True
        self._events.append(ProviderCallStarted(self.role, None, self._spec.clock()))
        try:
            if isinstance(self._spec, CameraWorkerSpec):
                camera = self._spec.factory(self._spec.configuration)
                self._camera = camera
                require_adapter_identity(camera.identity, self._spec.expected_identity, camera=True)
                observation = camera.open()
                require_camera_profile(observation, self._spec.expected_profile)
                priming = camera.next_frame()
                self._events.append(
                    CameraReady(
                        self.role,
                        priming.started_at,
                        priming.started_at,
                        priming.completed_at,
                        observation,
                        priming.read_id,
                        priming.started_at,
                        priming.completed_at,
                    )
                )
            else:
                joint = self._spec.factory(self._spec.configuration)
                self._joint = joint
                require_adapter_identity(joint.identity, self._spec.expected_identity, camera=False)
                joint.open()
                now = self._spec.clock()
                self._events.append(JointReady(self.role, now, now, now))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._events.append(
                ProviderFailed(
                    self.role,
                    "camera_readiness" if self.role is ProviderRole.CAMERA else "joint_connect",
                    None,
                    self._spec.clock(),
                    type(exc).__name__,
                    str(exc),
                    "fixture traceback",
                )
            )

    def send(self, command: ProviderCommand) -> None:
        match command:
            case StartProvider():
                self._start_provider()
            case ArmSample(sample_index=index):
                self._events.append(ProviderArmed(self.role, index, self._spec.clock()))
            case ReleaseSample(sample_index=index):
                self.release_count += 1
                self._events.append(ProviderCallStarted(self.role, index, self._spec.clock()))
                try:
                    if isinstance(self._spec, CameraWorkerSpec):
                        if self._camera is None:
                            raise RuntimeError("fake camera is unavailable")
                        read = self._camera.next_frame()
                        event = WorkerCompleted(self.role, index, read.completed_at, read, None)
                    else:
                        if self._joint is None:
                            raise RuntimeError("fake joint is unavailable")
                        read = self._joint.next_state()
                        event = WorkerCompleted(self.role, index, read.completed_at, None, read)
                    self._events.append(event)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    self._events.append(
                        ProviderFailed(
                            self.role,
                            "sample_pair",
                            index,
                            self._spec.clock(),
                            type(exc).__name__,
                            str(exc),
                            "fixture traceback",
                        )
                    )
            case StopProvider():
                cleanup_error: str | None = None
                try:
                    if self._camera is not None:
                        self._camera.close()
                    if self._joint is not None:
                        self._joint.close()
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    cleanup_error = f"{type(exc).__name__}: {exc}"
                self._alive = False
                self._events.append(
                    ProviderClosed(self.role, self._spec.clock(), True, cleanup_error)
                )

    def receive(self) -> ProviderEvent:
        return self._events.popleft()

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False

    def join(self, timeout: float) -> bool:
        del timeout
        self.join_calls += 1
        return not self._alive

    def exit_code(self) -> int | None:
        return 0 if not self._alive else None

    def child_failure(self) -> ProviderFailed | None:
        return None

    def close(self) -> None:
        return


class FakeProviderRuntime:
    """Event-driven fake runtime with no waits, sleeps, threads, or subprocesses."""

    def __init__(self) -> None:
        self.processes: list[FakeProviderProcess] = []

    def spawn(self, spec: WorkerSpec) -> ProviderProcess:
        process = FakeProviderProcess(spec)
        self.processes.append(process)
        return process

    def wait(
        self,
        processes: tuple[ProviderProcess, ...],
        timeout: float,
    ) -> tuple[ProviderProcess, ...]:
        del timeout
        return tuple(
            process
            for process in processes
            if isinstance(process, FakeProviderProcess) and process.has_event
        )
