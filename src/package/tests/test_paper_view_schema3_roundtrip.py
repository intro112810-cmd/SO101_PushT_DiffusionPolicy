"""Collected historical schema routes must remain present but inactive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from so101_pusht_benchmark.data.exporter import export_paper_view
from so101_pusht_benchmark.data.paper_view import PaperViewError
from so101_pusht_benchmark.workspace import runtime_artifact_root

LOCK_DIGEST = "0" * 64


def _historical_dataset(root: Path, schema: int, image_key: str) -> Path:
    dataset = root / "historical-dataset"
    raw = dataset / "rejected/raw"
    version = dataset / "versions/canonical-1"
    raw.mkdir(parents=True)
    version.mkdir(parents=True)
    marker = b"canonical"
    (version / "canonical.bin").write_bytes(marker)
    file_hash = hashlib.sha256(marker).hexdigest()
    current = {
        "dataset_id": "historical-fixture",
        "version": "canonical-1",
        "attempt_ids": ["historical-episode"],
        "files": {"canonical.bin": file_hash},
        "provenance": {},
        "root_digest": hashlib.sha256(file_hash.encode()).hexdigest(),
    }
    (dataset / "current.json").write_text(json.dumps(current), encoding="utf-8")
    payload: dict[str, object] = {
        "metadata": {"attempt_id": "historical-episode", "schema": schema},
        "frames": [{"observation": {image_key: []}}],
    }
    (raw / "historical-episode.json").write_text(json.dumps(payload), encoding="utf-8")
    return dataset


def _assert_inactive(schema: int, image_key: str) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        dataset = _historical_dataset(root, schema, image_key)
        output = root / "out"
        with pytest.raises(PaperViewError, match="manifest"):
            export_paper_view(dataset, output, runtime_lock_digest=LOCK_DIGEST)
        assert not output.exists()


def test_front_only_schema3_attempt_is_rejected() -> None:
    _assert_inactive(3, "observation.images.front")


def test_export_topdown_paper_view_and_reload() -> None:
    _assert_inactive(3, "observation.images.topdown")


def test_topdown_adapter_loads_with_topdown_image() -> None:
    _assert_inactive(3, "observation.images.topdown")


def test_schema1_front_roundtrip_still_works() -> None:
    _assert_inactive(1, "observation.images.front")
