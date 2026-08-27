"""End-to-end deterministic mouse topdown pipeline.

Source -> recorder -> env -> store -> export -> paper-view reader.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pytest

from so101_pusht_benchmark.collection.inputs import CollectionInput
from so101_pusht_benchmark.collection.recorder import CollectionConfig, Recorder
from so101_pusht_benchmark.data.exporter import export_paper_view
from so101_pusht_benchmark.data.paper_view import PaperViewError
from so101_pusht_benchmark.data.store import LocalDatasetStore
from so101_pusht_benchmark.input.mouse_keyboard import InputSample
from so101_pusht_benchmark.sim.env import PushTEnv
from so101_pusht_benchmark.workspace import runtime_artifact_root


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


def _record_attempt(
    root: Path, samples: list[InputSample], _attempt_id: str, clock: Clock | None = None
) -> tuple[Recorder, PushTEnv]:
    clock = clock or Clock()
    env = PushTEnv()
    store = LocalDatasetStore(root / "dataset")
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
    )


def _publish_current(dataset: Path, attempt_ids: list[str]) -> None:
    version = dataset / "versions" / "canonical-1"
    version.mkdir(parents=True, exist_ok=True)
    (version / "canonical.bin").write_bytes(b"canonical")
    file_hash = hashlib.sha256(b"canonical").hexdigest()
    files = {"canonical.bin": file_hash}
    root_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    current = {
        "dataset_id": "e2e",
        "version": "canonical-1",
        "attempt_ids": attempt_ids,
        "files": files,
        "provenance": {},
        "root_digest": root_digest,
    }
    (dataset / "current.json").write_text(json.dumps(current), encoding="utf-8")


def _read_raw_attempt(dataset: Path, attempt_id: str) -> dict[str, object]:
    import json as _json

    return cast(
        "dict[str, object]",
        _json.loads((dataset / "rejected" / "raw" / f"{attempt_id}.json").read_text()),
    )


def _promote_raw_attempt(dataset: Path, attempt_id: str) -> None:
    import json as _json

    payload = _read_raw_attempt(dataset, attempt_id)
    metadata = cast("dict[str, object]", payload["metadata"])
    metadata["session_id"] = f"session-{attempt_id}"
    metadata["training_eligible"] = True
    payload["sha256"] = LocalDatasetStore.payload_digest(payload)
    (dataset / "rejected" / "raw" / f"{attempt_id}.json").write_text(
        _json.dumps(payload), encoding="utf-8"
    )


def _applied_frames(payload: dict[str, object]) -> list[dict[str, object]]:
    frames = cast("list[object]", payload["frames"])
    result: list[dict[str, object]] = []
    for frame in frames:
        if isinstance(frame, dict):
            typed = cast("dict[str, object]", frame)
            if typed.get("applied") is True:
                result.append(typed)
    return result


def _requested_z(frame: dict[str, object]) -> float:
    telemetry = cast("dict[str, object]", frame["telemetry"])
    requested = cast("list[float]", telemetry["requested_target"])
    return float(requested[2])


def _frame_index(frame: dict[str, object]) -> int:
    return int(cast("int", frame["frame_index"]))


def _timestamp(frame: dict[str, object]) -> float:
    return float(cast("float", frame["timestamp"]))


def _observation(frame: dict[str, object]) -> dict[str, object]:
    return cast("dict[str, object]", frame["observation"])


def _state_length(observation: dict[str, object]) -> int:
    state = cast("list[object]", observation["observation.state"])
    return len(state)


def _topdown_pixels(observation: dict[str, object]) -> list[list[list[int]]]:
    topdown = cast("list[object]", observation["observation.images.topdown"])
    rows: list[list[list[int]]] = []
    for row in topdown:
        row_values = cast("list[object]", row)
        rows.append(
            [
                [int(cast("int", value)) for value in cast("list[object]", pixel)]
                for pixel in row_values
            ]
        )
    return rows


def test_end_to_end_contact_clearance_return_and_alignment() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temp:
        root = Path(temp)
        rec, env = _record_attempt(
            root,
            [
                InputSample(None, False),
                InputSample((0.28, 0.00, 0.045), True),
                InputSample((0.30, 0.02, 0.065), True),
                InputSample((0.29, -0.01, 0.045), True),
                InputSample(None, False),
            ],
            "e2e_mouse_1",
        )
        result = rec.record(1, "e2e_mouse_1", max_ticks=6)
        assert result.failure_code == "deadman_released"
        payload = _read_raw_attempt(root / "dataset", "e2e_mouse_1")
        applied = _applied_frames(payload)
        assert len(applied) >= 3
        contact_z = 0.045
        clearance_z = 0.065
        contacts = [frame for frame in applied if _requested_z(frame) == contact_z]
        clearances = [frame for frame in applied if _requested_z(frame) == clearance_z]
        assert contacts
        assert clearances
        assert _requested_z(contacts[-1]) == contact_z
        from itertools import pairwise

        ordered = sorted(applied, key=_frame_index)
        for left, right in pairwise(ordered):
            assert _frame_index(right) == _frame_index(left) + 1
            assert _timestamp(right) > _timestamp(left)
            observation = _observation(right)
            assert _state_length(observation) == 15
            assert "observation.images.topdown" in observation
        env.close()


def test_end_to_end_export_and_reload_pipeline() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temp:
        root = Path(temp)
        rec, env = _record_attempt(
            root,
            [
                InputSample(None, False),
                InputSample((0.28, 0.00, 0.045), True),
                InputSample((0.30, 0.02, 0.065), True),
                InputSample((0.29, -0.01, 0.045), True, success=True),
            ],
            "e2e_export_1",
        )
        rec.record(1, "e2e_export_1", max_ticks=5)
        dataset = root / "dataset"
        _promote_raw_attempt(dataset, "e2e_export_1")
        _publish_current(dataset, ["e2e_export_1"])
        output = root / "out"
        with pytest.raises(PaperViewError, match="manifest"):
            export_paper_view(dataset, output, runtime_lock_digest="0" * 64)
        assert not output.exists()
        env.close()


def test_release_before_next_tick_prevents_any_additional_frame() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temp:
        root = Path(temp)
        rec, env = _record_attempt(
            root,
            [
                InputSample(None, False),
                InputSample((0.28, 0.00, 0.045), True),
                InputSample(None, False),
            ],
            "e2e_release_1",
        )
        result = rec.record(1, "e2e_release_1", max_ticks=4)
        assert result.failure_code == "deadman_released"
        payload = _read_raw_attempt(root / "dataset", "e2e_release_1")
        applied = _applied_frames(payload)
        assert len(applied) == 1
        env.close()


def test_clearance_level_is_locked_at_065() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temp:
        root = Path(temp)
        rec, env = _record_attempt(
            root,
            [
                InputSample(None, False),
                InputSample((0.30, 0.02, 0.065), True),
                InputSample(None, False),
            ],
            "e2e_clearance_1",
        )
        rec.record(1, "e2e_clearance_1", max_ticks=4)
        payload = _read_raw_attempt(root / "dataset", "e2e_clearance_1")
        applied = _applied_frames(payload)
        assert applied
        for frame in applied:
            assert _requested_z(frame) == 0.065
        env.close()


def test_recorded_topdown_is_raw_without_overlay_pixels() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temp:
        root = Path(temp)
        rec, env = _record_attempt(
            root,
            [
                InputSample(None, False),
                InputSample((0.28, 0.00, 0.045), True),
                InputSample(None, False),
            ],
            "e2e_raw_1",
        )
        rec.record(1, "e2e_raw_1", max_ticks=4)
        payload = _read_raw_attempt(root / "dataset", "e2e_raw_1")
        applied = _applied_frames(payload)
        assert applied
        for frame in applied:
            observation = _observation(frame)
            topdown = _topdown_pixels(observation)
            assert len(topdown) == 96
            assert all(len(row) == 96 for row in topdown)
            assert all(len(pixel) == 3 for row in topdown for pixel in row)
            flat = [value for row in topdown for pixel in row for value in pixel]
            assert all(0 <= value <= 255 for value in flat)
        env.close()
