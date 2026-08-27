from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from so101_pusht_benchmark.data.exporter import export_paper_view
from so101_pusht_benchmark.data.importer import import_repo_store
from so101_pusht_benchmark.data.paper_view_reader import load_paper_view
from so101_pusht_benchmark.workspace import runtime_artifact_root

REPO_FEATURES = {
    "observation.images.cam_top": {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.cam_side": {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {"dtype": "float32", "shape": (5,)},
    "action": {"dtype": "float32", "shape": (2,)},
}


def create_mock_repo_store(root: Path, missing_key: str | None = None, n_frames: int = 3) -> Path:
    """Create one real LeRobot 0.4.4 native episode."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features: dict[str, object] = dict(REPO_FEATURES)
    if missing_key is not None:
        del features[missing_key]
    dataset = LeRobotDataset.create(
        "local/repo-fixture", fps=10, features=features, root=root, vcodec="h264"
    )
    for index in range(n_frames):
        frame: dict[str, object] = {
            "observation.images.cam_side": np.full((224, 224, 3), (255 - index) % 256, dtype=np.uint8),
            "observation.state": np.arange(5, dtype=np.float32) + index,
            "action": np.asarray([0.1, 0.2], dtype=np.float32),
            "task": "pushT",
        }
        if "observation.images.cam_top" in features:
            frame["observation.images.cam_top"] = np.full((224, 224, 3), index % 256, dtype=np.uint8)
        dataset.add_frame(frame)
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    return root


def test_import_repo_store_valid() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        repo = create_mock_repo_store(root / "repo")
        store = root / "native-store"
        assert import_repo_store(repo, store) == 0
        loaded = load_paper_view(store)
        assert loaded.manifest["episode_ids"] == ["0"]
        assert loaded.episode_ends.tolist() == [3]
        assert loaded.arrays["cam_top"].shape == (3, 224, 224, 3)
        assert loaded.arrays["cam_side"].shape == (3, 224, 224, 3)
        assert loaded.arrays["agent_pos"].shape == (3, 5)
        assert loaded.arrays["action"].shape == (3, 2)
        lock = str(loaded.manifest["runtime_lock_digest"])
        exported = export_paper_view(store, root / "export", runtime_lock_digest=lock)
        assert (
            load_paper_view(exported).manifest["canonical_digest"]
            == loaded.manifest["canonical_digest"]
        )
        assert not (store / "current.json").exists()
        assert not (store / "rejected/raw/ep0.json").exists()


def test_import_repo_store_missing_key() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        repo = create_mock_repo_store(root / "repo", "observation.images.cam_top")
        output = root / "native-store"
        assert import_repo_store(repo, output) != 0
        assert not output.exists()
        assert not list(root.glob(".native-store.tmp-*"))
        info = json.loads((repo / "meta/info.json").read_text(encoding="utf-8"))
        assert "observation.images.cam_top" not in info["features"]


def test_import_accepts_float32_timestamp_rounding() -> None:
    """LeRobot persists timestamps as float32; frame/FPS values above ~32 lose
    precision beyond the importer's 1e-6 abs_tol (e.g. 32.1 -> 32.099998).
    Import must accept real float32 rounding of the exact frame/FPS value."""
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        repo = create_mock_repo_store(root / "repo", n_frames=322)
        store = root / "native-store"
        assert import_repo_store(repo, store) == 0
        loaded = load_paper_view(store)
        assert loaded.manifest["episode_ids"] == ["0"]
        assert loaded.episode_ends.tolist() == [322]
