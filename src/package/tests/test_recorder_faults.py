"""Recorder must survive env fault StepResults (no loop_exception:KeyError).

``PushTEnv.step`` returns a fault StepResult (``info={"fault": ...}`` with no
``applied_target``) when the safety envelope is latched. The recorder loop
used to index ``out.info["applied_target"]`` unconditionally, so a fault
mid-take bubbled up as ``loop_exception:KeyError`` and lost the take.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from collection_fakes import RawStore

from so101_pusht_benchmark.collection.inputs import CollectionInput
from so101_pusht_benchmark.collection.recorder import AttemptResult, CollectionConfig, Recorder
from so101_pusht_benchmark.input.mouse_keyboard import InputSample
from so101_pusht_benchmark.sim.env import PushTEnv
from so101_pusht_benchmark.sim.safety import Fault


@dataclass
class Clock:
    now: float = 0.0

    def monotonic(self) -> float:
        return self.now


class LatchingMouse:
    """Polls samples, then latches the env safety before the deadman step.

    The first poll (neutral) arms the recorder; the second poll (deadman +
    valid target) triggers ``env.step`` while the safety envelope is already
    latched, so the step returns a fault StepResult.
    """

    def __init__(self, samples: list[InputSample], clock: Clock, env: PushTEnv) -> None:
        self.samples, self.clock, self.env = samples, clock, env
        self.polls = 0

    def poll(self) -> InputSample:
        value = self.samples.pop(0)
        self.polls += 1
        self.clock.now += 0.1
        if self.polls >= 2:
            self.env.safety.latch(Fault.FORBIDDEN_CONTACT)
        return value

    def close(self) -> None:
        pass


def run(tmp_path: Path) -> tuple[PushTEnv, RawStore, AttemptResult]:
    clock = Clock()
    env = PushTEnv()
    store = RawStore(tmp_path / "fault_collection")
    config = CollectionConfig(0.08, 0.012, 0.35, 2, 2, 0.005, 0.045, 0.050)
    adapter = CollectionInput.mouse(
        stale_timeout_s=0.35,
        debounce_ticks=2,
        contact_z_m=0.045,
        clearance_z_m=0.065,
        bounds_x=(0.18, 0.38),
        bounds_y=(-0.16, 0.16),
    )
    samples = [
        InputSample(None, False),
        InputSample((0.30, 0.0, 0.065), True),
        InputSample((0.30, 0.0, 0.065), True),
    ]
    mouse = LatchingMouse(samples, clock, env)
    rec = Recorder(env, mouse, store, config, clock, input_adapter=adapter)
    result = rec.record(3, "fault_attempt", max_ticks=3)
    return env, store, result


def test_env_fault_step_returns_fault_code_not_loop_exception(tmp_path: Path) -> None:
    env, store, result = run(tmp_path)
    try:
        assert result.failure_code != "loop_exception:KeyError"
        assert result.failure_code in (
            "terminal",
            "forbidden_contact",
            "environment_terminal",
        )
        assert store.attempts["fault_attempt"].metadata["failure_code"] == result.failure_code
    finally:
        env.close()


def test_env_fault_step_still_persists_attempt(tmp_path: Path) -> None:
    env, store, result = run(tmp_path)
    try:
        assert result.attempt_path is not None
        attempt = store.attempts["fault_attempt"]
        assert len(attempt.frames) >= 1
    finally:
        env.close()
