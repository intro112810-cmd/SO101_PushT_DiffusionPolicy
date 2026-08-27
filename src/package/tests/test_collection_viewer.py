from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from so101_pusht_benchmark import cli
from so101_pusht_benchmark.collection.recorder import AttemptResult
from so101_pusht_benchmark.collection.viewer import LiveViewer, RealtimePacer


@dataclass
class FakeClock:
    now: float = 0.0

    def monotonic(self) -> float:
        return self.now


def test_viewer_retains_the_unmodified_offscreen_rgb_frame() -> None:
    frame: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]] = np.zeros(
        (96, 96, 3), dtype=np.uint8
    )
    frame[0, 0, 0] = 1
    viewer = LiveViewer.open(enabled=False)
    # v1 (schema-1) callers pass a single frame, retained by identity.
    viewer.show(frame)
    assert not viewer.enabled
    assert viewer.last_frame is frame
    assert frame[0, 0, 0] != frame[0, 0, 1]
    viewer.close()


def test_viewer_composes_two_panes_without_mutating_sources() -> None:
    control = LiveViewer.open(enabled=False)
    observer = LiveViewer.open(enabled=False)
    paper: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]] = np.zeros(
        (96, 96, 3), dtype=np.uint8
    )
    front: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]] = np.zeros(
        (96, 96, 3), dtype=np.uint8
    )
    paper[0, 0, 0] = 1
    front[0, 0, 1] = 2
    paper.flags.writeable = False
    front.flags.writeable = False
    control.show(paper)
    observer.show(front)
    # Sources remain byte-identical; each window holds exactly its own frame.
    assert control.last_frame is paper
    assert observer.last_frame is front
    control.close()
    observer.close()


def test_realtime_pacer_uses_injected_clock_without_wall_clock_sleep() -> None:
    clock = FakeClock()
    delays: list[float] = []

    def sleep(delay: float) -> None:
        delays.append(delay)
        clock.now += delay

    pacer = RealtimePacer(clock, fps=10, sleep=sleep)
    pacer.wait()
    clock.now += 0.02
    pacer.wait()
    assert delays == [0.08]
    assert clock.now == 0.1


def test_collect_closes_viewer_source_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    class Source:
        def close(self) -> None:
            closed.append("source")

    class Environment:
        def close(self) -> None:
            closed.append("environment")

    class Viewer:
        @classmethod
        def open(cls, *, enabled: bool, title: str = "") -> Viewer:  # noqa: ARG003  # signature mirrors LiveViewer
            assert enabled
            return cls()

        def show(
            self,
            frame: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
            overlay: object = None,  # noqa: ARG002  # signature mirrors LiveViewer
            **_kwargs: object,
        ) -> None:
            assert frame.shape == (96, 96, 3)

        def close(self) -> None:
            closed.append("viewer")

    class Recording:
        def __init__(self, *_: object) -> None:
            pass

        def record(
            self,
            *_: object,
            before_tick: object,
            on_observation: object,
            **__: object,
        ) -> AttemptResult:
            assert callable(before_tick)
            assert callable(on_observation)
            before_tick()
            on_observation(np.zeros((96, 96, 3), dtype=np.uint8))
            return AttemptResult(False, "disconnect", 1, Path("attempt.json"))

    def store(_root: Path) -> object:
        return object()

    import so101_pusht_benchmark.collection.recorder as recorder_module
    import so101_pusht_benchmark.collection.viewer as viewer_module
    import so101_pusht_benchmark.data.store as store_module
    import so101_pusht_benchmark.integrations.lerobot.gamepad as gamepad_module
    import so101_pusht_benchmark.sim.env as sim_env

    monkeypatch.setattr(gamepad_module, "PublicGamepadSource", Source)
    monkeypatch.setattr(store_module, "LocalDatasetStore", store)
    monkeypatch.setattr(viewer_module, "LiveViewer", Viewer)
    monkeypatch.setattr(recorder_module, "Recorder", Recording)
    monkeypatch.setattr(sim_env, "PushTEnv", Environment)
    result = cli.main(
        [
            "collect-sim",
            "--root",
            "unused",
            "--seed",
            "0",
            "--attempt-id",
            "attempt_0",
            "--ticks",
            "1",
            "--max-attempts",
            "1",
        ]
    )
    assert result == 0
    assert closed == ["viewer", "source", "environment"]
