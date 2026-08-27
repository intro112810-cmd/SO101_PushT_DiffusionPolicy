from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from so101_pusht_benchmark.data.exporter import export_paper_view
from so101_pusht_benchmark.data.paper_view import (
    PaperArray,
    PaperViewError,
    PaperViewMetadata,
    canonical_digest,
    root_provenance_digest,
    trusted_runtime_lock_digest,
    write_paper_view,
)
from so101_pusht_benchmark.data.paper_view_reader import load_paper_view, validate_training_view
from so101_pusht_benchmark.data.splits import build_splits
from so101_pusht_benchmark.workspace import runtime_artifact_root

LOCK_DIGEST = trusted_runtime_lock_digest()


def _native_view(
    root: Path,
    episode_ids: list[str] | None = None,
    *,
    training_eligible: bool = False,
) -> Path:
    ids = episode_ids or ["z-last", "a-first"]
    rows_per_episode = 2
    row_count = len(ids) * rows_per_episode
    episode = np.repeat(np.arange(len(ids), dtype=np.int64), rows_per_episode)
    frame = np.tile(np.arange(rows_per_episode, dtype=np.int64), len(ids))
    ends = np.arange(rows_per_episode, row_count + 1, rows_per_episode, dtype=np.int64)
    arrays = {
        "cam_top": PaperArray(
            np.full((row_count, 224, 224, 3), 11, dtype=np.uint8), "rgb intensity"
        ),
        "cam_side": PaperArray(
            np.full((row_count, 224, 224, 3), 29, dtype=np.uint8), "rgb intensity"
        ),
        "agent_pos": PaperArray(np.full((row_count, 5), 0.5, dtype=np.float32), "radians"),
        "action": PaperArray(
            np.full((row_count, 2), 0.25, dtype=np.float32), "absolute normalized mocap XY"
        ),
        "timestamp": PaperArray(frame.astype(np.float64) / 10, "seconds"),
        "episode_id": PaperArray(episode, "episode ordinal"),
        "frame_index": PaperArray(frame, "frame ordinal"),
    }
    splits: dict[str, object] = {
        "frozen": training_eligible,
        "training_eligible": training_eligible,
        "train": ids if training_eligible else [],
        "validation": [],
        "test": [],
    }
    starts = [0, *ends.tolist()[:-1]]
    provenance: dict[str, object] = {
        "schema": "pusht-so100-root-provenance-v1",
        "source_members": {"fixture.bin": "1" * 64},
        "episodes": [
            {"episode_id": episode_id, "length": end - start}
            for episode_id, start, end in zip(ids, starts, ends.tolist(), strict=True)
        ],
    }
    metadata = PaperViewMetadata(
        canonical_digest(arrays, ends, ids),
        root_provenance_digest(provenance),
        provenance,
        ids,
        splits,
        LOCK_DIGEST,
        training_eligible,
    )
    return write_paper_view(root / "native", arrays, ends, metadata)


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _historical_store(root: Path) -> Path:
    source = root / "historical"
    (source / "rejected/raw").mkdir(parents=True)
    (source / "current.json").write_text(
        json.dumps({"attempt_ids": ["ep0"], "version": "canonical-import-1"}),
        encoding="utf-8",
    )
    return source


def test_empty_and_no_accepted_fail_closed() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        with pytest.raises(PaperViewError, match="real directory"):
            export_paper_view(root / "missing", root / "out", runtime_lock_digest=LOCK_DIGEST)
        with pytest.raises(PaperViewError, match="manifest"):
            export_paper_view(
                _historical_store(root), root / "out", runtime_lock_digest=LOCK_DIGEST
            )
        assert not (root / "out").exists()


def test_mixed_attempts_export_only_current_eligible_in_manifest_order() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        source = _native_view(root)
        loaded = load_paper_view(source)
        assert loaded.manifest["episode_ids"] == ["z-last", "a-first"]
        assert loaded.arrays["episode_id"].tolist() == [0, 0, 1, 1]
        assert loaded.arrays["frame_index"].tolist() == [0, 1, 0, 1]
        assert loaded.episode_ends.tolist() == [2, 4]


def test_synthetic_current_leak_rejected_and_test_view_is_training_ineligible() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        source = _native_view(Path(temporary))
        with pytest.raises(PaperViewError, match="training eligible"):
            validate_training_view(source)


def test_alignment_shapes_dtypes_units_and_applied_action() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        loaded = load_paper_view(_native_view(Path(temporary)))
        assert loaded.arrays["cam_top"].shape == (4, 224, 224, 3)
        assert loaded.arrays["cam_side"].shape == (4, 224, 224, 3)
        assert loaded.arrays["agent_pos"].shape == (4, 5)
        assert loaded.arrays["action"].shape == (4, 2)
        assert loaded.arrays["cam_top"].dtype == np.dtype(np.uint8)
        assert loaded.arrays["agent_pos"].dtype == np.dtype(np.float32)
        assert loaded.arrays["action"].dtype == np.dtype(np.float32)
        assert loaded.arrays["timestamp"].dtype == np.dtype(np.float64)


def test_double_export_is_byte_identical_and_digest_is_stable() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        source = _native_view(root)
        first = export_paper_view(source, root / "one", runtime_lock_digest=LOCK_DIGEST)
        second = export_paper_view(source, root / "two", runtime_lock_digest=LOCK_DIGEST)
        assert _tree(first) == _tree(second)
        digest = hashlib.sha256(
            b"".join(_tree(first)[name] for name in sorted(_tree(first)))
        ).hexdigest()
        assert len(digest) == 64


def test_tamper_mismatch_and_no_extra_members() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        source = _native_view(Path(temporary))
        chunk = source / "data/action/0.0"
        payload = chunk.read_bytes()
        chunk.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
        with pytest.raises(PaperViewError, match=r"\[-1,1\]|hash"):
            load_paper_view(source)


def test_split_quota_strata_and_session_disjointness() -> None:
    sessions = {f"episode-{index:03d}": f"session-{index // 20:02d}" for index in range(200)}
    splits = build_splits(sessions)
    assert {name: len(ids) for name, ids in splits.items()} == {
        "train": 160,
        "validation": 20,
        "test": 20,
    }
    sets = [{sessions[item] for item in splits[name]} for name in ("train", "validation", "test")]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])


def test_legacy_one_episode_split_is_not_training_eligible() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        source = _native_view(Path(temporary), ["episode-0"], training_eligible=True)
        with pytest.raises(PaperViewError, match="invalid frozen split manifest"):
            validate_training_view(source)
        assert not (source / "current.json").exists()


def test_paths_special_files_and_network_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        source = _native_view(root)
        link = root / "link"
        link.symlink_to(source, target_is_directory=True)
        with pytest.raises(PaperViewError, match=r"malformed|symlink"):
            export_paper_view(link, root / "out", runtime_lock_digest=LOCK_DIGEST)
        extra = source / "unexpected"
        extra.write_bytes(b"x")
        with pytest.raises(PaperViewError, match="membership"):
            load_paper_view(source)
        extra.unlink()

        def forbidden(*_args: object, **_kwargs: object) -> socket.socket:
            raise AssertionError("network attempted")

        monkeypatch.setattr(socket, "create_connection", forbidden)
        export_paper_view(source, root / "offline", runtime_lock_digest=LOCK_DIGEST)
        assert os.environ.get("HF_HUB_OFFLINE", "1") == "1"
