from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray
import pytest

import so101_pusht_benchmark.data.paper_view as paper_view_module

from so101_pusht_benchmark.data.paper_view import (
    PaperArray,
    PaperViewError,
    PaperViewMetadata,
    canonical_digest,
    root_provenance_digest,
    trusted_runtime_lock_digest,
    write_paper_view,
)
from so101_pusht_benchmark.data.paper_view_reader import load_paper_view
from so101_pusht_benchmark.workspace import runtime_artifact_root


def _arrays() -> dict[str, PaperArray]:
    frames = np.asarray([0, 1, 0, 1, 2], dtype=np.int64)
    episodes = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    return {
        "cam_top": PaperArray(np.full((5, 224, 224, 3), 17, dtype=np.uint8), "rgb intensity"),
        "cam_side": PaperArray(np.full((5, 224, 224, 3), 73, dtype=np.uint8), "rgb intensity"),
        "agent_pos": PaperArray(np.full((5, 5), 0.25, dtype=np.float32), "radians"),
        "action": PaperArray(
            np.asarray(
                [[-1.0, -0.75], [-0.5, -0.25], [0.0, 0.25], [0.5, 0.75], [1.0, 0.0]],
                dtype=np.float32,
            ),
            "absolute normalized mocap XY",
        ),
        "timestamp": PaperArray(frames.astype(np.float64) / 10, "seconds"),
        "episode_id": PaperArray(episodes, "episode ordinal"),
        "frame_index": PaperArray(frames, "frame ordinal"),
    }


def _metadata(
    ids: list[str] | None = None,
    arrays: dict[str, PaperArray] | None = None,
    ends: NDArray[np.int64] | None = None,
) -> PaperViewMetadata:
    episode_ids = ids or ["0", "1"]
    values = arrays or _arrays()
    episode_ends = ends if ends is not None else np.asarray([2, 5], dtype=np.int64)
    starts = [0, *episode_ends.tolist()[:-1]]
    provenance: dict[str, object] = {
        "schema": "pusht-so100-root-provenance-v1",
        "source_members": {"fixture.bin": "4" * 64},
        "episodes": [
            {"episode_id": episode_id, "length": end - start}
            for episode_id, start, end in zip(
                episode_ids, starts, episode_ends.tolist(), strict=True
            )
        ],
    }
    return PaperViewMetadata(
        canonical_digest(values, episode_ends, episode_ids),
        root_provenance_digest(provenance),
        provenance,
        episode_ids,
        {"frozen": False, "training_eligible": False, "train": [], "validation": [], "test": []},
        trusted_runtime_lock_digest(),
        False,
    )


def test_native_paper_view_roundtrip_preserves_exact_arrays() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        destination = Path(temporary) / "view"
        expected = _arrays()
        loaded = load_paper_view(
            write_paper_view(destination, expected, np.asarray([2, 5], dtype=np.int64), _metadata())
        )
        for name, array in expected.items():
            assert loaded.arrays[name].tobytes() == array.values.tobytes()
            assert loaded.arrays[name].dtype == array.values.dtype
        assert loaded.episode_ends.tolist() == [2, 5]
        assert loaded.manifest["fps"] == 10


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("extra", "exact policy"),
        ("unequal", "shape or dtype"),
        ("nonfinite", "non-finite"),
        ("nonchronological", "chronological"),
        ("timestamp", "frame_index/10"),
    ],
)
def test_writer_rejects_malformed_native_arrays_atomically(mutation: str, error: str) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        destination = root / "view"
        arrays = _arrays()
        if mutation == "extra":
            arrays["velocity"] = PaperArray(np.zeros((5, 5), dtype=np.float32), "forbidden")
        elif mutation == "unequal":
            arrays["cam_side"] = PaperArray(
                np.zeros((4, 224, 224, 3), dtype=np.uint8), "rgb intensity"
            )
        elif mutation == "nonfinite":
            arrays["action"].values[0, 0] = np.nan
        elif mutation == "nonchronological":
            arrays["frame_index"].values[3:] = [2, 1]
        else:
            arrays["timestamp"].values[4] = 0.25
        ends = np.asarray([2, 5], dtype=np.int64)
        with pytest.raises(PaperViewError, match=error):
            write_paper_view(destination, arrays, ends, _metadata(arrays=arrays, ends=ends))
        assert not destination.exists()
        assert not list(root.glob(".view.tmp-*"))


def test_writer_rejects_duplicate_episode_ids_atomically() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        with pytest.raises(PaperViewError, match="duplicate"):
            write_paper_view(
                root / "view",
                _arrays(),
                np.asarray([2, 5], dtype=np.int64),
                _metadata(["same", "same"], _arrays(), np.asarray([2, 5], dtype=np.int64)),
            )
        assert not (root / "view").exists()


def test_interrupted_write_cleans_partial_state_and_can_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        destination = root / "view"
        original = paper_view_module.write_array_chunks
        calls = 0

        def interrupt(path: Path, array: NDArray[np.generic], rows: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            original(path, array, rows)

        monkeypatch.setattr(paper_view_module, "write_array_chunks", interrupt)
        with pytest.raises(KeyboardInterrupt):
            write_paper_view(
                destination, _arrays(), np.asarray([2, 5], dtype=np.int64), _metadata()
            )
        assert not destination.exists()
        assert not list(root.glob(".view.tmp-*"))
        monkeypatch.setattr(paper_view_module, "write_array_chunks", original)
        assert (
            write_paper_view(
                destination, _arrays(), np.asarray([2, 5], dtype=np.int64), _metadata()
            )
            == destination
        )


def test_repeated_write_preserves_immutable_store_without_temporary_state() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        destination = write_paper_view(
            root / "view", _arrays(), np.asarray([2, 5], dtype=np.int64), _metadata()
        )
        before = {
            path.relative_to(destination).as_posix(): path.read_bytes()
            for path in destination.rglob("*")
            if path.is_file()
        }
        with pytest.raises(FileExistsError, match="already exists"):
            write_paper_view(
                destination, _arrays(), np.asarray([2, 5], dtype=np.int64), _metadata()
            )
        after = {
            path.relative_to(destination).as_posix(): path.read_bytes()
            for path in destination.rglob("*")
            if path.is_file()
        }
        assert before == after
        assert not list(root.glob(".view.tmp-*"))


def test_action_out_of_range_fails_before_publication() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        arrays = _arrays()
        arrays["action"].values[0, 0] = np.float32(2.0)
        ends = np.asarray([2, 5], dtype=np.int64)
        with pytest.raises(PaperViewError, match=r"\[-1,1\]"):
            write_paper_view(root / "view", arrays, ends, _metadata(arrays=arrays, ends=ends))
        assert not (root / "view").exists()


@pytest.mark.parametrize(("field", "value"), [("root", "short"), ("runtime", "F" * 64)])
def test_writer_rejects_malformed_provenance_hashes(field: str, value: str) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        arrays, ends = _arrays(), np.asarray([2, 5], dtype=np.int64)
        valid = _metadata(arrays=arrays, ends=ends)
        metadata = PaperViewMetadata(
            valid.canonical_digest,
            value if field == "root" else valid.root_digest,
            valid.root_provenance,
            valid.episode_ids,
            valid.splits,
            value if field == "runtime" else valid.runtime_lock_digest,
            False,
        )
        with pytest.raises(PaperViewError, match="lowercase SHA-256"):
            write_paper_view(root / "view", arrays, ends, metadata)
        assert not (root / "view").exists()


def test_writer_rejects_forged_root_digest_and_runtime_lock() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        arrays, ends = _arrays(), np.asarray([2, 5], dtype=np.int64)
        valid = _metadata(arrays=arrays, ends=ends)
        for root_digest, runtime_digest, expected in (
            ("f" * 64, valid.runtime_lock_digest, "root provenance digest mismatch"),
            (valid.root_digest, "f" * 64, "trusted runtime lock digest mismatch"),
        ):
            forged = PaperViewMetadata(
                valid.canonical_digest,
                root_digest,
                valid.root_provenance,
                valid.episode_ids,
                valid.splits,
                runtime_digest,
                False,
            )
            with pytest.raises(PaperViewError, match=expected):
                write_paper_view(root / expected.split()[0], arrays, ends, forged)


@pytest.mark.parametrize(
    "provenance",
    [
        {},
        {
            "schema": "pusht-so100-root-provenance-v1",
            "source_members": {"fixture.bin": "4" * 64},
            "episodes": [
                {"episode_id": "0", "length": 2},
                {"episode_id": "1", "length": 3},
            ],
            "unknown": True,
        },
    ],
)
def test_writer_rejects_missing_or_unknown_root_provenance(
    provenance: dict[str, object],
) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        arrays, ends = _arrays(), np.asarray([2, 5], dtype=np.int64)
        valid = _metadata(arrays=arrays, ends=ends)
        malformed = PaperViewMetadata(
            valid.canonical_digest,
            "f" * 64,
            provenance,
            valid.episode_ids,
            valid.splits,
            valid.runtime_lock_digest,
            False,
        )
        with pytest.raises(PaperViewError, match="root provenance"):
            write_paper_view(Path(temporary) / "view", arrays, ends, malformed)


def test_writer_rejects_forged_canonical_digest() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        arrays, ends = _arrays(), np.asarray([2, 5], dtype=np.int64)
        valid = _metadata(arrays=arrays, ends=ends)
        forged = PaperViewMetadata(
            "f" * 64,
            valid.root_digest,
            valid.root_provenance,
            valid.episode_ids,
            valid.splits,
            valid.runtime_lock_digest,
            False,
        )
        with pytest.raises(PaperViewError, match="canonical digest mismatch"):
            write_paper_view(root / "view", arrays, ends, forged)


@pytest.mark.parametrize("field", ["root_digest", "runtime_lock_digest"])
def test_reader_rejects_forged_root_or_runtime_identity(field: str) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        destination = write_paper_view(
            root / "view", _arrays(), np.asarray([2, 5], dtype=np.int64), _metadata()
        )
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest[field] = "f" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        expected = (
            "root provenance digest mismatch"
            if field == "root_digest"
            else "trusted runtime lock digest mismatch"
        )
        with pytest.raises(PaperViewError, match=expected):
            load_paper_view(destination)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_reader_rejects_missing_or_unknown_root_provenance(mutation: str) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        destination = write_paper_view(
            root / "view", _arrays(), np.asarray([2, 5], dtype=np.int64), _metadata()
        )
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "missing":
            del manifest["root_provenance"]
        else:
            manifest["root_provenance"]["unknown"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        expected = "manifest keys" if mutation == "missing" else "root provenance"
        with pytest.raises(PaperViewError, match=expected):
            load_paper_view(destination)


@pytest.mark.parametrize("forged", ["short", "f" * 64])
def test_reader_rejects_malformed_or_forged_canonical_digest(forged: str) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        destination = write_paper_view(
            root / "view", _arrays(), np.asarray([2, 5], dtype=np.int64), _metadata()
        )
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["canonical_digest"] = forged
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        expected = "lowercase SHA-256" if forged == "short" else "canonical digest mismatch"
        with pytest.raises(PaperViewError, match=expected):
            load_paper_view(destination)


def test_reader_recomputes_canonical_digest_after_episode_id_tamper() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        destination = write_paper_view(
            root / "view", _arrays(), np.asarray([2, 5], dtype=np.int64), _metadata()
        )
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["episode_ids"] = ["forged-0", "forged-1"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(PaperViewError, match=r"canonical digest|root provenance"):
            load_paper_view(destination)


def test_reader_rejects_episode_boundary_byte_tamper() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        destination = write_paper_view(
            root / "view", _arrays(), np.asarray([2, 5], dtype=np.int64), _metadata()
        )
        chunk = destination / "episode_ends/0"
        values = np.frombuffer(chunk.read_bytes(), dtype=np.int64).copy()
        values[0] = 1
        chunk.write_bytes(values.tobytes())
        with pytest.raises(PaperViewError, match=r"alignment|hash|boundaries|root provenance"):
            load_paper_view(destination)


def test_foreign_staging_fails_closed_and_is_not_deleted() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        destination = root / "view"
        foreign = root / ".view.tmp-999999"
        foreign.mkdir()
        marker = foreign / "partial"
        marker.write_bytes(b"foreign")
        with pytest.raises(PaperViewError, match="staging"):
            write_paper_view(
                destination, _arrays(), np.asarray([2, 5], dtype=np.int64), _metadata()
            )
        assert not destination.exists()
        assert marker.read_bytes() == b"foreign"


def test_reader_rejects_malformed_or_extra_manifest_metadata() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        destination = write_paper_view(
            root / "view", _arrays(), np.asarray([2, 5], dtype=np.int64), _metadata()
        )
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["selected_view"] = "top"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(PaperViewError, match="manifest keys"):
            load_paper_view(destination)
