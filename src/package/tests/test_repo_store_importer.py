from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from native_dataset_fixture import create_two_episode_repo
from so101_pusht_benchmark.data.importer import import_repo_store
from so101_pusht_benchmark.data.paper_view_reader import load_paper_view
from so101_pusht_benchmark.workspace import runtime_artifact_root


def _root() -> TemporaryDirectory[str]:
    return TemporaryDirectory(dir=runtime_artifact_root())


def _assert_atomic_failure(repo: Path, destination: Path) -> None:
    assert import_repo_store(repo, destination) != 0
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.tmp-*"))


def _rewrite_info(repo: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    path = repo / "meta/info.json"
    value = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _rewrite_parquet(path: Path, column: str, values: pa.Array) -> None:
    table = pq.read_table(path)
    index = table.column_names.index(column)
    pq.write_table(table.set_column(index, column, values), path)


def test_importer_enumerates_all_native_episodes_without_projection() -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        destination = root / "native-store"

        assert import_repo_store(repo, destination) == 0
        loaded = load_paper_view(destination)

        assert loaded.manifest["episode_ids"] == ["0", "1"]
        assert loaded.episode_ends.tolist() == [2, 5]
        assert set(loaded.arrays) == {
            "cam_top",
            "cam_side",
            "agent_pos",
            "action",
            "timestamp",
            "episode_id",
            "frame_index",
        }
        assert loaded.arrays["cam_top"].shape == (5, 224, 224, 3)
        assert loaded.arrays["cam_side"].shape == (5, 224, 224, 3)
        assert loaded.arrays["cam_top"].dtype == np.dtype(np.uint8)
        assert loaded.arrays["cam_side"].dtype == np.dtype(np.uint8)
        assert loaded.arrays["agent_pos"].shape == (5, 5)
        assert loaded.arrays["agent_pos"].dtype == np.dtype(np.float32)
        assert loaded.arrays["action"].shape == (5, 2)
        assert loaded.arrays["action"].dtype == np.dtype(np.float32)
        assert loaded.arrays["timestamp"].tolist() == [0.0, 0.1, 0.0, 0.1, 0.2]
        assert loaded.arrays["episode_id"].tolist() == [0, 0, 1, 1, 1]
        assert loaded.arrays["frame_index"].tolist() == [0, 1, 0, 1, 2]


@pytest.mark.parametrize("camera", ["cam_top", "cam_side"])
def test_missing_camera_fails_without_store_or_temporary(camera: str) -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        video = next((repo / f"videos/observation.images.{camera}").glob("chunk-*/file-*.mp4"))
        video.unlink()
        _assert_atomic_failure(repo, root / "output")


def test_short_camera_fails_atomically() -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        episode_path = next((repo / "meta/episodes").glob("chunk-*/file-*.parquet"))
        name = "videos/observation.images.cam_side/to_timestamp"
        _rewrite_parquet(episode_path, name, pa.array([0.2, 0.6], type=pa.float64()))
        _assert_atomic_failure(repo, root / "output")


@pytest.mark.parametrize("fps", [9, 11])
def test_wrong_metadata_fps_fails_atomically(fps: int) -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        _rewrite_info(repo, lambda value: value.__setitem__("fps", fps))
        _assert_atomic_failure(repo, root / "output")


@pytest.mark.parametrize(("column", "width"), [("observation.state", 5), ("action", 2)])
def test_nonfinite_policy_values_fail_atomically(column: str, width: int) -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        data_path = next((repo / "data").glob("chunk-*/file-*.parquet"))
        table = pq.read_table(data_path)
        values = cast("list[list[float]]", table[column].to_pylist())
        values[2][0] = float("nan")
        array = pa.array(values, type=pa.list_(pa.float32(), width))
        _rewrite_parquet(data_path, column, array)
        _assert_atomic_failure(repo, root / "output")


def test_out_of_range_source_action_fails_atomically() -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        data_path = next((repo / "data").glob("chunk-*/file-*.parquet"))
        table = pq.read_table(data_path)
        values = cast("list[list[float]]", table["action"].to_pylist())
        values[0][0] = 2.0
        _rewrite_parquet(data_path, "action", pa.array(values, type=pa.list_(pa.float32(), 2)))
        _assert_atomic_failure(repo, root / "output")


def test_duplicate_episode_ids_fail_atomically() -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        path = next((repo / "meta/episodes").glob("chunk-*/file-*.parquet"))
        _rewrite_parquet(path, "episode_index", pa.array([0, 0], type=pa.int64()))
        _assert_atomic_failure(repo, root / "output")


def test_partial_episode_fails_atomically() -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        path = next((repo / "meta/episodes").glob("chunk-*/file-*.parquet"))
        _rewrite_parquet(path, "length", pa.array([2, 2], type=pa.int64()))
        _assert_atomic_failure(repo, root / "output")


def test_nonchronological_frames_fail_atomically() -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        path = next((repo / "data").glob("chunk-*/file-*.parquet"))
        _rewrite_parquet(path, "frame_index", pa.array([0, 1, 0, 2, 1], type=pa.int64()))
        _assert_atomic_failure(repo, root / "output")


def test_extra_policy_key_fails_atomically() -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")

        def add_extra(value: dict[str, object]) -> None:
            features = cast("dict[str, object]", value["features"])
            features["observation.velocity"] = {"dtype": "float32", "shape": [5]}

        _rewrite_info(repo, add_extra)
        _assert_atomic_failure(repo, root / "output")


def test_importer_rejects_foreign_staging_without_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        output = root / "output"
        foreign = root / ".output.tmp-777777"
        foreign.mkdir()
        marker = foreign / "partial"
        marker.write_bytes(b"foreign")
        assert import_repo_store(repo, output) != 0
        captured = capsys.readouterr()
        assert '"store"' not in captured.out
        assert "staging" in captured.err
        assert not output.exists()
        assert marker.read_bytes() == b"foreign"


def test_malformed_metadata_fails_atomically() -> None:
    with _root() as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        (repo / "meta/info.json").write_text("{broken", encoding="utf-8")
        _assert_atomic_failure(repo, root / "output")
