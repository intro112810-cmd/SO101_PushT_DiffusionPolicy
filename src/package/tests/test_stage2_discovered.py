from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from typing import TypedDict, cast

import pytest

from so101_pusht_benchmark.data.store import digest_tree
from so101_pusht_benchmark.data.validator import qualify_attempt, verify_current
from so101_pusht_benchmark.data.splits import SplitError, build_splits
from so101_pusht_benchmark.workspace import WorkspacePolicyError


class Telemetry(TypedDict, total=False):
    frame_id: int
    ack_status: str
    applied_action: list[float]
    coverage: float
    max_coverage: float
    observation_rgb_hash: str
    observation_state_hash: str
    action_hash: str
    timestamp: float


class Frame(TypedDict):
    frame_index: int
    timestamp: float
    applied: bool
    telemetry: Telemetry
    observation: dict[str, object]


class AttemptMetadata(TypedDict, total=False):
    task: str
    mode: str
    physical_device: bool
    device_provenance: dict[str, object]
    synthetic: bool
    attempt_id: str


def _frame(index: int, *, flags: dict[str, object] | None = None) -> Frame:
    telemetry: Telemetry = {
        "frame_id": index,
        "ack_status": "applied",
        "applied_action": [0.25, 0.0, 0.05],
        "coverage": 0.95,
        "max_coverage": 0.95,
    }
    telemetry.update(cast("Telemetry", flags or {}))
    return {
        "frame_index": index,
        "timestamp": index / 10,
        "applied": True,
        "telemetry": telemetry,
        "observation": {},
    }


def _attempt(*, flags: dict[str, object] | None = None) -> dict[str, object]:
    metadata: AttemptMetadata = {
        "task": "push_t",
        "mode": "human_gamepad",
        "physical_device": True,
        "device_provenance": {"adapter": "lerobot_public_gamepad", "physical": True},
    }
    return {
        "metadata": metadata,
        "frames": [_frame(0, flags=flags)],
    }


def test_tree_walk_accepts_directories_and_rejects_special_entries(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "frame.bin").write_bytes(b"x")
    digest, files = digest_tree(tmp_path)
    assert files == {"data/frame.bin": __import__("hashlib").sha256(b"x").hexdigest()}
    assert digest
    (tmp_path / "link").symlink_to(tmp_path / "data" / "frame.bin")
    with pytest.raises(WorkspacePolicyError):
        digest_tree(tmp_path)


def test_tree_walk_rejects_fifo_socket_and_device(tmp_path: Path) -> None:
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(WorkspacePolicyError):
        digest_tree(tmp_path)
    fifo.unlink()
    server = socket.socket(socket.AF_UNIX)
    try:
        server.bind(str(tmp_path / "sock"))
        with pytest.raises(WorkspacePolicyError):
            digest_tree(tmp_path)
    finally:
        server.close()


def test_qualification_requires_task_rejects_synthetic_and_integrity_flags() -> None:
    missing = _attempt()
    missing_metadata = missing["metadata"]
    assert isinstance(missing_metadata, dict)
    del missing_metadata["task"]
    assert not qualify_attempt(missing)[0]
    synthetic = _attempt()
    synthetic_metadata = synthetic["metadata"]
    assert isinstance(synthetic_metadata, dict)
    synthetic_metadata["synthetic"] = True
    assert "synthetic_attempt" in qualify_attempt(synthetic)[1]
    for flag in (
        "dropped",
        "duplicate",
        "clipped",
        "forbidden_contact",
        "incomplete_media",
        "replay_mismatch",
    ):
        assert not qualify_attempt(_attempt(flags={flag: True}))[0]


def test_qualification_requires_both_max_and_final_coverage() -> None:
    attempt = _attempt()
    frames = attempt["frames"]
    assert isinstance(frames, list)
    frame = cast("Frame", frames[0])
    telemetry = frame["telemetry"]
    telemetry["coverage"] = 0.94
    telemetry["max_coverage"] = 0.96
    assert "coverage_not_met" in qualify_attempt(attempt)[1]


def test_allowed_contact_is_not_forbidden() -> None:
    attempt = _attempt(flags={"contact": True, "contact_allowed": True})
    assert qualify_attempt(attempt)[0]


def test_split_requires_exact_quota_and_session_disjoint_ids() -> None:
    episodes = {f"ep-{i:03d}": f"session-{i // 20}" for i in range(200)}
    split = build_splits(episodes)
    assert {key: len(value) for key, value in split.items()} == {
        "train": 160,
        "validation": 20,
        "test": 20,
    }
    with pytest.raises(SplitError):
        build_splits(dict(list(episodes.items())[:-1]))


def test_current_manifest_rejects_extra_file_and_root_digest_tamper(tmp_path: Path) -> None:
    version = tmp_path / "versions" / "v"
    version.mkdir(parents=True)
    (version / "a").write_bytes(b"a")
    digest = "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
    files = {"a": digest}
    root_digest = (
        __import__("hashlib")
        .sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())
        .hexdigest()
    )
    (tmp_path / "current.json").write_text(
        json.dumps({"version": "v", "files": files, "root_digest": root_digest})
    )
    assert verify_current(tmp_path) == (True, None)
    (version / "extra").write_bytes(b"x")
    assert verify_current(tmp_path)[0] is False
    (version / "extra").unlink()
    (tmp_path / "current.json").write_text(
        json.dumps({"version": "v", "files": files, "root_digest": "bad"})
    )
    assert verify_current(tmp_path)[1] == "canonical_tree_digest_mismatch"
