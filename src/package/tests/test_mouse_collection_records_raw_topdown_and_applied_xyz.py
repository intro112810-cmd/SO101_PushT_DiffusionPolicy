"""Schema-3 mouse collection records raw topdown observations and applied XYZ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from collection_fakes import RawStore

from so101_pusht_benchmark.collection.inputs import CollectionInput
from so101_pusht_benchmark.collection.recorder import CollectionConfig, Recorder
from so101_pusht_benchmark.input.mouse_keyboard import InputSample
from so101_pusht_benchmark.sim.env import PushTEnv


@dataclass
class Clock:
    now: float = 0.0

    def monotonic(self) -> float:
        return self.now


class SyntheticMouse:
    def __init__(self, samples: list[InputSample], clock: Clock, advance: float = 0.1) -> None:
        self.samples, self.clock, self.advance = samples, clock, advance

    def poll(self) -> InputSample:
        if not self.samples:
            return InputSample(None, False)
        value = self.samples.pop(0)
        self.clock.now += self.advance
        return value

    def close(self) -> None:
        pass


def recorder(
    tmp_path: Path, samples: list[InputSample], clock: Clock | None = None
) -> tuple[Recorder, PushTEnv, RawStore]:
    clock = clock or Clock()
    env = PushTEnv()
    store = RawStore(tmp_path / "mouse_collection")
    config = CollectionConfig(0.08, 0.012, 0.35, 2, 2, 0.005, 0.045, 0.050)
    adapter = CollectionInput.mouse(
        stale_timeout_s=0.35,
        debounce_ticks=2,
        contact_z_m=0.045,
        clearance_z_m=0.065,
        bounds_x=(0.18, 0.38),
        bounds_y=(-0.16, 0.16),
    )
    return (
        Recorder(env, SyntheticMouse(samples, clock), store, config, clock, input_adapter=adapter),
        env,
        store,
    )


def test_mouse_collection_records_raw_topdown_and_applied_xyz(tmp_path: Path) -> None:
    rec, env, store = recorder(
        tmp_path,
        [
            InputSample(None, False),
            InputSample((0.28, 0.00, 0.045), True),
            InputSample((0.30, 0.02, 0.065), True),
            InputSample(None, False),
        ],
    )
    result = rec.record(1, "mouse_topdown_1", max_ticks=5)
    assert result.failure_code == "deadman_released"
    assert result.frames >= 2
    attempt = store.attempts["mouse_topdown_1"]
    metadata = attempt.metadata
    assert metadata["mode"] == "human_mouse_keyboard"
    assert metadata["schema"] == 3
    assert metadata["physical_device"] is False
    provenance = cast("dict[str, object]", metadata["device_provenance"])
    assert provenance["adapter"] == "mouse_keyboard_topdown_v3"

    applied_frames = [frame for frame in attempt.frames if frame.applied]
    assert len(applied_frames) >= 2
    first = applied_frames[0]
    assert set(first.observation) == {"observation.images.topdown", "observation.state"}
    assert first.observation["observation.images.topdown"].dtype == "uint8"
    assert first.observation["observation.images.topdown"].shape == (96, 96, 3)
    assert first.observation["observation.state"].dtype == "float32"
    assert first.observation["observation.state"].shape == (15,)
    assert isinstance(first.action, tuple)
    assert len(first.action) == 3
    request = first.requested_target
    assert request[0] >= 0.18
    assert request[0] <= 0.38
    assert request[1] >= -0.16
    assert request[1] <= 0.16
    assert request[2] >= 0.045
    assert request[2] <= 0.065

    second = applied_frames[1]
    request2 = second.requested_target
    assert request2[2] == 0.065
    env.close()


def test_mouse_invalid_target_is_rejected_fail_closed(tmp_path: Path) -> None:
    rec, env, store = recorder(
        tmp_path,
        [
            InputSample(None, False),
            InputSample((0.50, 0.30, 0.045), True),
        ],
    )
    result = rec.record(1, "mouse_outside_1", max_ticks=3)
    assert result.failure_code == "invalid_target"
    attempt = store.attempts["mouse_outside_1"]
    applied_frames = [frame for frame in attempt.frames if frame.applied]
    assert applied_frames == []
    env.close()


def test_mouse_idle_without_deadman_keeps_window_open(tmp_path: Path) -> None:
    rec, env, store = recorder(
        tmp_path,
        # Neutral input only: the window must stay open (no deadman fault)
        # until the operator actually engages, ending only via coverage.
        [InputSample(None, False), InputSample(None, False), InputSample(None, False)],
    )
    result = rec.record(1, "mouse_idle_1", max_ticks=3)
    attempt = store.attempts["mouse_idle_1"]
    assert attempt.metadata["synthetic"] is False
    assert attempt.metadata["physical_device"] is False
    assert result.failure_code == "coverage_not_met"
    assert result.frames == 0
    applied_frames = [frame for frame in attempt.frames if frame.applied]
    assert applied_frames == []
    env.close()


def test_mouse_loop_exception_persists_raw_telemetry_not_sample(tmp_path: Path) -> None:
    """The exception/interrupt fault path must never store the polled sample.

    The persisted attempt stays JSON-serializable.
    """

    class ExplodingSource:
        def __init__(self) -> None:
            self.armed = True

        def poll(self) -> InputSample:
            if self.armed:
                self.armed = False
                return InputSample((0.28, 0.0, 0.045), True)
            raise RuntimeError("boom")

        def close(self) -> None:
            pass

    clock = Clock()
    env = PushTEnv()
    store = RawStore(tmp_path / "mouse_exception")
    config = CollectionConfig(0.08, 0.012, 0.35, 2, 2, 0.005, 0.045, 0.050)
    adapter = CollectionInput.mouse(
        stale_timeout_s=0.35,
        debounce_ticks=2,
        contact_z_m=0.045,
        clearance_z_m=0.065,
        bounds_x=(0.18, 0.38),
        bounds_y=(-0.16, 0.16),
    )
    rec = Recorder(
        env,
        ExplodingSource(),
        store,
        config,
        clock,
        input_adapter=adapter,
    )
    result = rec.record(1, "mouse_exception_1", max_ticks=4)
    assert result.failure_code == "loop_exception:RuntimeError"
    attempt = store.attempts["mouse_exception_1"]
    # Every stored raw_axes must be a JSON scalar, never the InputSample.
    for frame in attempt.frames:
        raw = frame.raw_axes
        assert raw is None or isinstance(raw, tuple)
    env.close()
