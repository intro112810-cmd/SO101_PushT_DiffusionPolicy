from __future__ import annotations

from pathlib import Path

import pytest

from so101_pusht_benchmark.sim_to_real import preview
from so101_pusht_benchmark.sim_to_real.preview import (
    render_prediction_preview,
)


def test_preview_renders_all_predicted_targets_without_actuation(
    tmp_path: Path,
) -> None:
    actions = [[0.25 + index * 0.002, 0.01 - index * 0.002] for index in range(8)]
    png = tmp_path / "preview.png"
    mp4 = tmp_path / "preview.mp4"

    result = render_prediction_preview(
        actions,
        png_path=png,
        mp4_path=mp4,
        seed=100018,
        evidence_scope="test_fixture_only",
    )

    assert result["predicted_target_count"] == 8
    assert result["evidence_scope"] == "test_fixture_only"
    assert result["frame_count"] == 9
    assert result["actuation_performed"] is False
    assert result["motor_writes_performed"] is False
    assert png.stat().st_size > 1_000
    assert mp4.stat().st_size > 1_000


def test_preview_failure_removes_partial_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    png = tmp_path / "partial.png"
    mp4 = tmp_path / "partial.mp4"

    def fail_video(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("fixture encoder failure")

    monkeypatch.setattr(preview.imageio, "mimsave", fail_video)
    with pytest.raises(RuntimeError, match="encoder failure"):
        render_prediction_preview(
            [[0.25, 0.01] for _ in range(8)],
            png_path=png,
            mp4_path=mp4,
            seed=100018,
            evidence_scope="test_fixture_only",
        )

    assert not png.exists()
    assert not mp4.exists()
