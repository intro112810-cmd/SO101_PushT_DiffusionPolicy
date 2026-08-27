from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from so101_pusht_benchmark.data.importer import import_repo_store
from so101_pusht_benchmark.data.paper_view_reader import load_paper_view
from so101_pusht_benchmark.native_cli import main as native_main
from so101_pusht_benchmark.workspace import runtime_artifact_root

from test_importer import create_mock_repo_store


def _historical_export_config(root: Path, selected_view: str) -> Path:
    path = root / f"paper_view_{selected_view}.yaml"
    path.write_text(
        "\n".join(
            [
                "schema: 1",
                "exporter_revision: paper_view_v1",
                f"selected_view: {selected_view}",
                "runtime_lock: environments/sim-runtime.lock",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_dual_cam_importer_writes_both_views() -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        root = Path(temporary)
        repo = create_mock_repo_store(root / "repo")
        output = root / "native-view"
        assert import_repo_store(repo, output) == 0
        loaded = load_paper_view(output)
        assert loaded.arrays["cam_top"].shape == (3, 224, 224, 3)
        assert loaded.arrays["cam_side"].shape == (3, 224, 224, 3)
        assert loaded.arrays["cam_top"].dtype == np.dtype(np.uint8)
        assert loaded.arrays["cam_side"].dtype == np.dtype(np.uint8)
        assert loaded.arrays["cam_top"].tobytes() != loaded.arrays["cam_side"].tobytes()


@pytest.mark.parametrize("selected_view", ["side", "top", "bogus"])
def test_historical_selected_view_export_is_inactive(selected_view: str) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        config = _historical_export_config(Path(temporary), selected_view)
        assert native_main(["export-native", "--preflight", "--config", str(config)]) == 1
