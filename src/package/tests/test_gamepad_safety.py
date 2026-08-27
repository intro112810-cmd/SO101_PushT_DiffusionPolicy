from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from collection_fakes import RawStore

import pytest

from so101_pusht_benchmark.collection.recorder import CollectionConfig, Recorder, RecorderState
from so101_pusht_benchmark.integrations.lerobot.gamepad import GamepadSample
from so101_pusht_benchmark.sim.env import PushTEnv


@dataclass
class Clock:
    now: float = 0.0

    def monotonic(self) -> float:
        return self.now


class Events:
    def __init__(self, samples: list[GamepadSample], clock: Clock, advance: float = 0.1) -> None:
        self.samples, self.clock, self.advance = samples, clock, advance

    def poll(self) -> GamepadSample:
        value = self.samples.pop(0)
        self.clock.now += self.advance
        return value

    def close(self) -> None:
        pass


def recorder(
    tmp_path: Path, samples: list[GamepadSample], clock: Clock | None = None
) -> tuple[Recorder, PushTEnv]:
    clock = clock or Clock()
    env = PushTEnv()
    store = RawStore(tmp_path / "collection")
    config = CollectionConfig(0.08, 0.012, 0.35, 2, 2, 0.005, 0.045, 0.050)
    return Recorder(env, Events(samples, clock), store, config, clock), env


def test_disconnected_requires_neutral_then_deadman_arms(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="XYZ 3-tuple"):
        GamepadSample(cast("tuple[float, float, float]", (0.0, 0.0)), False)
    rec, env = recorder(
        tmp_path,
        [
            GamepadSample((0.7, 0.0, 0.0), True, connected=True),
            GamepadSample((0.0, 0.0, 0.0), False),
            GamepadSample((0.5, 0.0, 1.0), True),
            GamepadSample((0.0, 0.0, 1.0), True),
        ],
    )
    result = rec.record(1, "attempt_state", max_ticks=4)
    assert rec.state is RecorderState.ARMED
    assert result.failure_code == "coverage_not_met"
    assert result.frames == 2
    assert [frame.action[2] for frame in rec.last_frames] == [0.05, 0.05]
    assert [frame.requested_target[2] for frame in rec.last_frames] == pytest.approx([0.05, 0.05])
    env.close()


@pytest.mark.parametrize(
    ("sample", "reason"),
    [
        (GamepadSample((0.0, 0.0, 0.0), False), "deadman_released"),
        (GamepadSample((0.0, 0.0, 0.0), True, connected=False), "disconnect"),
    ],
)
def test_deadman_release_and_disconnect_abort_before_substeps(
    tmp_path: Path, sample: GamepadSample, reason: str
) -> None:
    rec, env = recorder(
        tmp_path,
        [
            GamepadSample((0.0, 0.0, 0.0), False),
            GamepadSample((0.1, 0.0, 0.0), True),
            sample,
        ],
    )
    result = rec.record(1, "attempt_abort", max_ticks=3)
    assert result.failure_code == reason
    assert not env.active
    assert not env.safety.safe
    assert (env.scene.data.ctrl == env.scene.data.qpos[: env.scene.data.ctrl.size]).all()
    assert result.frames == 2
    env.close()


def test_stale_and_software_stop_latch_environment_safety(tmp_path: Path) -> None:
    clock = Clock()
    rec, env = recorder(
        tmp_path,
        [GamepadSample((0.0, 0.0, 0.0), False), GamepadSample((0.1, 0.0, 0.0), True)],
        clock,
    )
    rec.source = Events(
        [
            GamepadSample((0.0, 0.0, 0.0), False),
            GamepadSample((0.1, 0.0, 0.0), True, fresh=False),
        ],
        clock,
        1.0,
    )
    stale = rec.record(1, "attempt_stale", max_ticks=2)
    assert stale.failure_code == "stale_input"
    assert not env.safety.safe
    env.close()
    rec, env = recorder(
        tmp_path,
        [
            GamepadSample((0.0, 0.0, 0.0), False),
            GamepadSample((0.0, 0.0, 0.0), True, stop=True),
        ],
    )
    stopped = rec.record(1, "attempt_stop", max_ticks=2)
    assert stopped.failure_code == "software_stop"
    assert rec.state is RecorderState.STOPPED
    assert "physical" not in stopped.failure_code
    env.close()
