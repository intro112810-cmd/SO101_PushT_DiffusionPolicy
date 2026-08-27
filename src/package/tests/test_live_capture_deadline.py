"""Exact sample-pair completion deadline classification."""

from __future__ import annotations

from collections import deque

import pytest

from so101_pusht_benchmark.sim_to_real.live_capture_pair import ReadPairRequest, read_pair
from so101_pusht_benchmark.sim_to_real.live_capture_phase import PhaseFaultError
from so101_pusht_benchmark.sim_to_real.live_capture_protocol import (
    ArmSample,
    ProviderArmed,
    ProviderCallStarted,
    ProviderCommand,
    ProviderEvent,
    ProviderFailed,
    ProviderProcess,
    ProviderRole,
    ReleaseSample,
    StartProvider,
    StopProvider,
    WorkerCompleted,
    WorkerSpec,
)
from so101_pusht_benchmark.sim_to_real.live_capture_types import (
    TimedCameraRead,
    TimedJointRead,
)


class _DeadlineProcess:
    """Minimal scripted process for exact completion-boundary classification."""

    def __init__(self, role: ProviderRole, completed_at: float) -> None:
        self.role = role
        self._completed_at = completed_at
        self._events: deque[ProviderEvent] = deque()

    @property
    def has_event(self) -> bool:
        return bool(self._events)

    def start(self) -> None:
        return

    def send(self, command: ProviderCommand) -> None:
        match command:
            case ArmSample(sample_index=index):
                self._events.append(ProviderArmed(self.role, index, 10.0))
            case ReleaseSample(sample_index=index):
                self._events.append(ProviderCallStarted(self.role, index, 10.0))
                if self.role is ProviderRole.CAMERA:
                    camera = TimedCameraRead(
                        "camera-000",
                        b"frame",
                        self._completed_at,
                        self._completed_at,
                    )
                    joint = None
                else:
                    camera = None
                    joint = TimedJointRead(
                        "joint-000",
                        (0.0, 1.0, 2.0, 3.0, 4.0),
                        self._completed_at,
                        self._completed_at,
                    )
                self._events.append(
                    WorkerCompleted(
                        self.role,
                        index,
                        self._completed_at,
                        camera,
                        joint,
                    )
                )
            case StartProvider() | StopProvider():
                return

    def receive(self) -> ProviderEvent:
        return self._events.popleft()

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        return

    def kill(self) -> None:
        return

    def join(self, timeout: float) -> bool:
        del timeout
        return True

    def exit_code(self) -> int | None:
        return 0

    def child_failure(self) -> ProviderFailed | None:
        return None

    def close(self) -> None:
        return


class _DeadlineRuntime:
    def spawn(self, spec: WorkerSpec) -> ProviderProcess:
        del spec
        raise AssertionError("deadline test does not spawn")

    def wait(
        self,
        processes: tuple[ProviderProcess, ...],
        timeout: float,
    ) -> tuple[ProviderProcess, ...]:
        del timeout
        return tuple(
            process
            for process in processes
            if isinstance(process, _DeadlineProcess) and process.has_event
        )


@pytest.mark.parametrize("offset", [0.0, 0.000001])
def test_pair_completion_exact_deadline_is_on_time_and_above_is_quarantined(
    offset: float,
) -> None:
    processes = (
        _DeadlineProcess(ProviderRole.CAMERA, 10.2 + offset),
        _DeadlineProcess(ProviderRole.JOINT, 10.2 + offset),
    )
    request = ReadPairRequest(
        _DeadlineRuntime(),
        processes,
        0,
        0.2,
        lambda: 10.0,
        [],
    )

    if offset == 0.0:
        camera, joint, phase = read_pair(request)
        assert camera.completed_at == 10.2
        assert joint.completed_at == 10.2
        assert phase.deadline == 10.2
    else:
        with pytest.raises(PhaseFaultError, match="after its deadline") as caught:
            read_pair(request)
        assert caught.value.primary.phase == "sample_pair"
        assert caught.value.primary.observed_at > caught.value.phase_evidence.deadline
