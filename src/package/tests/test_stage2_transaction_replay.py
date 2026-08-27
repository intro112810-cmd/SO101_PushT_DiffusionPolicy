from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from so101_pusht_benchmark.data.store import FrameRecord, LocalDatasetStore, PublishRequest
from so101_pusht_benchmark.data.validator import replay_attempt
from so101_pusht_benchmark.workspace import runtime_artifact_root


FEATURES: dict[str, object] = {
    "observation.images.front": {"dtype": "uint8", "shape": [96, 96, 3]},
    "observation.state": {"dtype": "float32", "shape": [15]},
    "action": {"dtype": "float32", "shape": [3]},
}


class Telemetry(TypedDict, total=False):
    frame_id: int
    ack_status: str
    applied_action: list[float]
    observation_rgb_hash: str
    observation_state_hash: str
    action_hash: str


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def replay_payload() -> dict[str, object]:
    rgb = [[[0, 0, 0]]]
    state = [0.0] * 15
    action = [0.25, 0.0, 0.05]
    telemetry: Telemetry = {
        "frame_id": 0,
        "ack_status": "applied",
        "applied_action": action,
        "observation_rgb_hash": digest(rgb),
        "observation_state_hash": digest(state),
        "action_hash": digest(action),
    }
    return {
        "metadata": {"attempt_id": "replay0", "task": "push_t"},
        "frames": [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "action": action,
                "observation": {"observation.images.front": rgb, "observation.state": state},
                "telemetry": telemetry,
                "applied": True,
            }
        ],
    }


def _write_replay(root: Path, payload: dict[str, object]) -> None:
    raw = root / "rejected" / "raw"
    raw.mkdir(parents=True)
    encoded = {**payload, "sha256": ""}
    encoded["sha256"] = LocalDatasetStore.payload_digest(encoded)
    (raw / "replay0.json").write_text(json.dumps(encoded))


def test_replay_accepts_exact_rgb_state_action_timestamp_trace() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temp:
        root = Path(temp)
        _write_replay(root, replay_payload())
        assert replay_attempt(root, "replay0") == {"replay_match": True, "frames": 1, "errors": []}


def test_replay_rejects_rgb_state_action_and_timestamp_mismatch() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temp:
        root = Path(temp)
        payload = replay_payload()
        frame = cast("dict[str, object]", cast("list[object]", payload["frames"])[0])
        observation = cast("dict[str, object]", frame["observation"])
        telemetry = cast("dict[str, object]", frame["telemetry"])
        observation["observation.images.front"] = [[[1, 0, 0]]]
        observation["observation.state"] = [1.0] * 15
        frame["action"] = [0.2, 0.0, 0.05]
        frame["timestamp"] = 0.2
        _write_replay(root, payload)
        result = replay_attempt(root, "replay0")
        assert result["replay_match"] is False
        assert set(cast("list[str]", result["errors"])) >= {
            "timing_mismatch",
            "applied_action_mismatch",
            "observation_rgb_mismatch",
            "observation_state_mismatch",
            "action_hash_mismatch",
        }
        assert telemetry["frame_id"] == 0


class FakeDataset:
    def __init__(self, root: Path) -> None:
        root.mkdir()
        self.root = root

    def add_frame(self, frame: dict[str, object]) -> None:
        assert frame["task"] == "push_t"
        (self.root / "frame.json").write_text(json.dumps({"task": frame["task"]}))

    def save_episode(self) -> None:
        (self.root / "episode.json").write_text("{}")

    def finalize(self) -> None:
        (self.root / "meta").mkdir()
        (self.root / "meta" / "done").write_text("ok")


def maker(**kwargs: object) -> FakeDataset:
    return FakeDataset(cast(Path, kwargs["root"]))


def request(attempt_id: str, fault: str | None = None) -> PublishRequest:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    state = np.zeros(15, dtype=np.float32)
    frame = FrameRecord(
        0,
        0.0,
        {"observation.images.front": image, "observation.state": state},
        (0.25, 0.0, 0.05),
        None,
        (0.25, 0.0, 0.05),
        {},
        {"observation.images.front": image, "observation.state": state},
        True,
    )

    def crash(phase: str) -> None:
        if phase == fault:
            raise RuntimeError(phase)

    return PublishRequest(
        attempt_id,
        [frame],
        {
            "task": "push_t",
            "mode": "human_gamepad",
            "physical_device": True,
            "training_eligible": True,
        },
        FEATURES,
        create=maker,
        reload=lambda _: None,
        fault=crash,
    )


def test_offline_reload_missing_local_fails_without_socket_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temp:
        missing = Path(temp) / "missing"

        def forbidden(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("network fallback attempted")

        monkeypatch.setattr("socket.create_connection", forbidden)
        with pytest.raises(FileNotFoundError):
            LocalDatasetStore.reload_local(missing)


def test_transaction_crash_recovery_quarantines_orphan_and_preserves_current() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temp:
        store = LocalDatasetStore(Path(temp) / "dataset")
        store.publish_human_episode(request("good"))
        before = (store.root / "current.json").read_bytes()
        for phase in ("after_save_episode", "after_finalize", "before_version_rename"):
            with pytest.raises(RuntimeError):
                store.publish_human_episode(request(phase, phase))
            recovered = store.recover()
            assert phase in recovered["orphaned"]
            assert (store.root / "current.json").read_bytes() == before
        assert not list((store.root / "staging").glob("*"))
        assert (store.root / "rejected" / "staging").is_dir()
