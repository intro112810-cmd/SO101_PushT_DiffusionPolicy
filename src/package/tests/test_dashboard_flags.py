"""Track failed-episode markers for the advanced dashboard (scripts/experiment_dashboard.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from scripts import experiment_dashboard  # noqa: E402


@pytest.fixture
def flags_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "episode_flags.json"
    monkeypatch.setattr(experiment_dashboard, "FLAGS_PATH", path)
    return path


def _flag_ids(result: dict) -> list[int]:
    return [int(row["episode_id"]) for row in result["flags"]]


def test_mark_and_read_failed_episodes(flags_path: Path) -> None:
    result = experiment_dashboard.set_failed_flag(3, True)
    assert result["ok"] is True
    result = experiment_dashboard.set_failed_flag(7, True)
    assert _flag_ids(result) == [3, 7]
    assert flags_path.exists()


def test_clear_failed_marker(flags_path: Path) -> None:
    experiment_dashboard.set_failed_flag(3, True)
    result = experiment_dashboard.set_failed_flag(3, False)
    assert result["flags"] == []
    assert flags_path.exists()  # clear persists an empty store


def test_reject_invalid_episode_ids(flags_path: Path) -> None:
    assert experiment_dashboard.set_failed_flag(-1, True)["ok"] is False
    assert experiment_dashboard.set_failed_flag(None, True)["ok"] is False
    assert not flags_path.exists()  # invalid calls never create the store


def test_markers_persist_to_disk(flags_path: Path) -> None:
    experiment_dashboard.set_failed_flag(5, True)
    assert '"5"' in flags_path.read_text(encoding="utf-8")
    assert _flag_ids(experiment_dashboard.read_flags()) == [5]


def test_tolerates_corrupted_store(flags_path: Path) -> None:
    flags_path.write_text("{not-json", encoding="utf-8")
    assert experiment_dashboard.read_flags()["flags"] == []
    experiment_dashboard.set_failed_flag(2, True)
    assert _flag_ids(experiment_dashboard.read_flags()) == [2]


def test_adjust_flags_after_delete_renumbers(flags_path: Path) -> None:
    for episode_id in (1, 5, 9):
        experiment_dashboard.set_failed_flag(episode_id, True)
    experiment_dashboard.adjust_flags_after_delete(5)
    assert _flag_ids(experiment_dashboard.read_flags()) == [1, 8]
    assert flags_path.exists()  # renumbered markers persist to disk
