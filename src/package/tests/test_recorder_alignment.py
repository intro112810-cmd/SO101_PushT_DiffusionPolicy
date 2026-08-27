from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from collection_fakes import Attempt, RawStore

from so101_pusht_benchmark.collection.recorder import CollectionConfig, Recorder
from so101_pusht_benchmark.integrations.lerobot.gamepad import GamepadSample
from so101_pusht_benchmark.sim.env import PushTEnv


@dataclass
class Clock:
    now: float = 0.0

    def monotonic(self) -> float:
        return self.now


class Events:
    def __init__(self, values: list[GamepadSample], clock: Clock) -> None:
        self.values, self.clock = values, clock

    def poll(self) -> GamepadSample:
        self.clock.now += 0.1
        return self.values.pop(0)

    def close(self) -> None:
        pass


def run(
    tmp_path: Path,
    values: list[GamepadSample],
    name: str = "attempt_alignment",
    replacement_for: str | None = None,
) -> Attempt:
    clock = Clock()
    env = PushTEnv()
    store = RawStore(tmp_path / "collection")
    _result = Recorder(
        env,
        Events(values, clock),
        store,
        CollectionConfig(0.08, 0.012, 0.35, 2, 2, 0.005, 0.045, 0.050),
        clock,
    ).record(2, name, replacement_for=replacement_for, max_ticks=len(values))
    attempt = store.attempts[name]
    env.close()
    return attempt


def test_frame_records_raw_requested_applied_and_aligned_observations(tmp_path: Path) -> None:
    payload = run(
        tmp_path,
        [
            GamepadSample((0.0, 0.0, 0.0), False),
            GamepadSample((1.0, 1.0, 1.0), True),
            GamepadSample((1.0, 1.0, 1.0), True),
        ],
    )
    frames = payload.frames
    assert [frame.timestamp for frame in frames] == [0.0, 0.1]
    first = frames[0].telemetry
    assert first["raw_axes"] == (1.0, 1.0, 1.0)
    assert len(cast("tuple[float, float, float]", first["requested_target"])) == 3
    assert len(cast("tuple[float, float, float]", first["applied_action"])) == 3
    assert first["requested_target"] != first["applied_target"]
    assert first["action_timestamp"] == 0.0
    assert first["next_state_timestamp"] == 0.1
    assert isinstance(first["command_id"], int)
    assert isinstance(first["frame_id"], int)
    assert first["command_id"] < first["frame_id"]
    assert isinstance(first["ik_joint_target"], list)
    assert isinstance(first["ctrl_command"], list)
    ik_target = cast("list[object]", first["ik_joint_target"])
    ctrl_command = cast("list[object]", first["ctrl_command"])
    assert len(ik_target) == 6
    assert len(ctrl_command) == 6


def test_success_is_request_not_acceptance_and_buttons_conflict_debounce(tmp_path: Path) -> None:
    payload = run(
        tmp_path,
        [
            GamepadSample((0.0, 0.0, 0.0), False),
            GamepadSample((0.0, 0.0, 0.0), True, success=True),
            GamepadSample((0.0, 0.0, 0.0), True, success=True),
        ],
        "attempt_success",
    )
    assert payload.metadata["operator_success_requested"] is True
    assert payload.metadata["training_eligible"] is False
    conflict = run(
        tmp_path,
        [
            GamepadSample((0.0, 0.0, 0.0), False),
            GamepadSample((0.0, 0.0, 0.0), True, success=True, rerecord=True),
        ],
        "attempt_conflict",
    )
    assert conflict.metadata["failure_code"] == "button_conflict"


def test_rerecord_preserves_attempt_and_links_replacement(tmp_path: Path) -> None:
    old = run(
        tmp_path,
        [
            GamepadSample((0.0, 0.0, 0.0), False),
            GamepadSample((0.0, 0.0, 0.0), True, rerecord=True),
        ],
        "attempt_old",
    )
    replacement = run(
        tmp_path,
        [GamepadSample((0.0, 0.0, 0.0), False)],
        "attempt_new",
        "attempt_old",
    )
    assert old.metadata["failure_code"] == "rerecord"
    assert replacement.metadata["replacement_for"] == "attempt_old"
