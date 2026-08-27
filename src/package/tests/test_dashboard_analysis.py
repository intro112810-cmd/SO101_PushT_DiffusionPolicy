"""Training-curve parsing, eval comparison fields, and dataset analysis for the dashboard."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from scripts import experiment_dashboard  # noqa: E402


@pytest.fixture
def fake_art_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "art"
    monkeypatch.setattr(experiment_dashboard, "ART_ROOT", root)
    return root


@pytest.fixture
def fake_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "dataset"
    monkeypatch.setattr(experiment_dashboard.SERVER, "dataset_root", root, raising=False)
    return root


def _write_logs(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _points(curves: dict, model: str) -> list[dict]:
    return curves[model]


def test_training_curves_parses_valid_jsonl(fake_art_root: Path) -> None:
    rows = [
        {"train_loss": 1.2, "global_step": 0, "epoch": 0, "lr": 1e-6},
        {"train_loss": 0.8, "global_step": 1, "epoch": 0, "lr": 2e-6},
        {"train_loss": 0.6, "global_step": 2, "epoch": 1, "lr": 3e-6},
    ]
    _write_logs(fake_art_root / "models" / "dp_cnn" / "full" / "logs.json.txt", rows)
    result = experiment_dashboard.read_training_curves()
    assert result["ok"] is True
    points = _points(result["curves"], "dp_cnn")
    assert [point["step"] for point in points] == [0, 1, 2]
    assert points[0]["loss"] == pytest.approx(1.2)
    assert points[0]["lr"] == pytest.approx(1e-6)
    assert result["curves"]["dp_transformer"] == []


def test_training_curves_tolerates_malformed_lines(fake_art_root: Path) -> None:
    log = fake_art_root / "models" / "ibc" / "full" / "logs.json.txt"
    _write_logs(
        log,
        [
            {"train_loss": 1.0, "global_step": 0},
            {"train_loss": 0.5, "global_step": 1},
        ],
    )
    log.write_text(
        '{"train_loss": 1.0, "global_step": 0}\nnot-json-line\n'
        '{"train_loss": 0.5, "global_step": 1}\n',
        encoding="utf-8",
    )
    points = _points(experiment_dashboard.read_training_curves()["curves"], "ibc")
    assert [point["step"] for point in points] == [0, 1]


def test_training_curves_picks_newest_file(fake_art_root: Path) -> None:
    old = fake_art_root / "models" / "lstm_gmm" / "full" / "logs.json.txt"
    new = fake_art_root / "models" / "lstm_gmm" / "run-2" / "logs.json.txt"
    _write_logs(old, [{"train_loss": 9.9, "global_step": 0}])
    _write_logs(new, [{"train_loss": 0.1, "global_step": 0}, {"train_loss": 0.05, "global_step": 1}])
    os.utime(new, (1000, 2000))
    os.utime(old, (1000, 1000))
    points = _points(experiment_dashboard.read_training_curves()["curves"], "lstm_gmm")
    assert points[0]["loss"] == pytest.approx(0.1)  # newest file wins


def test_training_curves_downsampling_preserves_endpoints(fake_art_root: Path) -> None:
    rows = [{"train_loss": float(i), "global_step": i} for i in range(600)]
    _write_logs(fake_art_root / "models" / "dp_transformer" / "full" / "logs.json.txt", rows)
    points = _points(experiment_dashboard.read_training_curves()["curves"], "dp_transformer")
    assert len(points) <= experiment_dashboard.MAX_CURVE_POINTS
    assert points[0]["step"] == 0
    assert points[-1]["step"] == 599


def test_training_curves_empty_when_no_logs(fake_art_root: Path) -> None:
    assert not (fake_art_root / "models").exists()  # precondition: no model dirs at all
    result = experiment_dashboard.read_training_curves()
    assert all(result["curves"][model] == [] for model in experiment_dashboard.MODELS)


def test_dataset_analysis_reads_parquet(fake_dataset: Path) -> None:
    chunk = fake_dataset / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1, 1],
            "action": [[0.1, 0.2], [0.3, 0.4], [-0.5, 0.6], [0.7, -0.8], [0.9, 1.0]],
        }
    )
    frame.to_parquet(chunk / "file-000.parquet", index=False)
    result = experiment_dashboard.read_dataset_analysis()
    assert result["readable"] is True
    assert result["lengths"] == [2, 3]
    assert result["mean_length"] == pytest.approx(2.5)
    assert result["action"]["x"]["min"] == pytest.approx(-0.5)
    assert result["action"]["x"]["max"] == pytest.approx(0.9)
    assert result["action"]["x"]["n"] == 5


def test_dataset_analysis_empty_series_without_action(fake_dataset: Path) -> None:
    chunk = fake_dataset / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"episode_index": [0, 0, 1]})
    frame.to_parquet(chunk / "file-000.parquet", index=False)
    result = experiment_dashboard.read_dataset_analysis()
    assert result["readable"] is True
    assert result["action"]["x"] == {"min": None, "max": None, "mean": None, "std": None, "n": 0}


def test_dataset_analysis_unreadable_parquet(fake_dataset: Path) -> None:
    chunk = fake_dataset / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    (chunk / "file-000.parquet").write_bytes(b"not a parquet file at all")
    result = experiment_dashboard.read_dataset_analysis()
    assert result["ok"] is True
    assert result["readable"] is False
    assert result["note"]


def test_evaluation_chart_fields_from_fixture(fake_art_root: Path) -> None:
    report = fake_art_root / "reports" / "four-model-final"
    report.mkdir(parents=True, exist_ok=True)
    (report / "comparison.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "dp_cnn": {
                        "eval/success_rate": 0.8,
                        "eval/mean_steps": 120.0,
                        "eval/mean_dxy": 0.01,
                        "eval/mean_dyaw": 1.5,
                        "eval/mean_duration_s": 12.0,
                    },
                    "ibc": {
                        "eval/success_rate": 0.5,
                        "eval/mean_steps": 200.0,
                        "eval/mean_dxy": 0.02,
                        "eval/mean_dyaw": 2.0,
                        "eval/mean_duration_s": 20.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    result = experiment_dashboard.read_evaluation()
    assert result["ok"] is True
    rows = {row["model"]: row for row in result["rows"]}
    assert rows["dp_cnn"]["success_rate"] == pytest.approx(0.8)
    assert rows["ibc"]["mean_steps"] == pytest.approx(200.0)


def test_evaluation_empty_when_report_absent(fake_art_root: Path) -> None:
    report = fake_art_root / "reports" / "four-model-final" / "comparison.json"
    assert not report.exists()  # precondition: no report written yet
    result = experiment_dashboard.read_evaluation()
    assert result["ok"] is False
