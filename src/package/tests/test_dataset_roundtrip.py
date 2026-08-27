from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq
import pytest

from native_dataset_fixture import create_two_episode_repo
from so101_pusht_benchmark.data.exporter import ExportError, export_paper_view
from so101_pusht_benchmark.data.importer import decode_video, import_repo_store
from so101_pusht_benchmark.data.paper_view_reader import load_paper_view
from so101_pusht_benchmark.workspace import runtime_artifact_root


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(files[name])
    return digest.hexdigest()


def test_real_two_episode_import_export_reader_roundtrip_is_digest_stable() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        store = root / "native-store"
        assert import_repo_store(repo, store) == 0
        imported = load_paper_view(store)

        source_top = decode_video(
            next((repo / "videos/observation.images.cam_top").glob("chunk-*/file-*.mp4"))
        )
        source_side = decode_video(
            next((repo / "videos/observation.images.cam_side").glob("chunk-*/file-*.mp4"))
        )
        table = pq.read_table(next((repo / "data").glob("chunk-*/file-*.parquet")))
        assert imported.arrays["cam_top"].tobytes() == source_top.tobytes()
        assert imported.arrays["cam_side"].tobytes() == source_side.tobytes()
        expected_state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        expected_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        assert imported.arrays["agent_pos"].tobytes() == expected_state.tobytes()
        assert imported.arrays["action"].tobytes() == expected_action.tobytes()

        lock_digest = str(imported.manifest["runtime_lock_digest"])
        first = export_paper_view(store, root / "first", runtime_lock_digest=lock_digest)
        second = export_paper_view(store, root / "second", runtime_lock_digest=lock_digest)
        first_files, second_files = _files(first), _files(second)
        assert first_files == second_files
        assert _digest(first_files) == _digest(second_files)

        reloaded = load_paper_view(first)
        assert reloaded.manifest["episode_ids"] == ["0", "1"]
        assert reloaded.episode_ends.tolist() == [2, 5]
        for name in imported.arrays:
            assert reloaded.arrays[name].tobytes() == imported.arrays[name].tobytes()
            assert reloaded.arrays[name].dtype == imported.arrays[name].dtype


def test_export_rejects_tampered_camera_without_partial_output() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        store = root / "native-store"
        assert import_repo_store(repo, store) == 0
        manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
        lock_digest = manifest["runtime_lock_digest"]
        chunk = store / "data/cam_side/0.0.0.0"
        payload = chunk.read_bytes()
        chunk.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
        output = root / "export"
        with pytest.raises(Exception, match=r"hash|metadata"):
            export_paper_view(store, output, runtime_lock_digest=lock_digest)
        assert not output.exists()
        assert not list(root.glob(".*.tmp-*"))


def test_export_cannot_launder_forged_canonical_digest() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        store = root / "native-store"
        assert import_repo_store(repo, store) == 0
        manifest_path = store / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock_digest = manifest["runtime_lock_digest"]
        manifest["canonical_digest"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        output = root / "export"
        with pytest.raises(Exception, match="canonical digest mismatch"):
            export_paper_view(store, output, runtime_lock_digest=lock_digest)
        assert not output.exists()


def test_export_rejects_forged_root_digest_before_output() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        store = root / "native-store"
        assert import_repo_store(repo, store) == 0
        manifest_path = store / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock_digest = manifest["runtime_lock_digest"]
        manifest["root_digest"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        output = root / "export"
        with pytest.raises(Exception, match="root provenance digest mismatch"):
            export_paper_view(store, output, runtime_lock_digest=lock_digest)
        assert not output.exists()


def test_export_rejects_runtime_digest_mismatch_before_output() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        repo = create_two_episode_repo(root / "repo")
        store = root / "native-store"
        assert import_repo_store(repo, store) == 0
        output = root / "export"
        with pytest.raises(ExportError, match="trusted runtime lock digest mismatch"):
            export_paper_view(store, output, runtime_lock_digest="f" * 64)
        assert not output.exists()
