"""Per-episode final-success metrics (video-based dxy estimate) for the dashboard.

The recorder never persists its live "Match dxy" value, so the dashboard
recovers a per-episode success estimate from the recorded top-view video:
the green puck's centroid distance to the fixed goal centroid. These tests
pin the estimation pipeline, its caching, and its invalidation on dataset
changes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

BENCH_ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from scripts import experiment_dashboard  # noqa: E402


@pytest.fixture
def fake_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "dataset"
    monkeypatch.setattr(experiment_dashboard.SERVER, "dataset_root", root, raising=False)
    return root


def _write_fake_parquet(root: Path, lengths: list[int]) -> None:
    rows: list[dict[str, object]] = []
    for episode_id, frames in enumerate(lengths):
        for frame_index in range(frames):
            rows.append(
                {
                    "episode_index": episode_id,
                    "frame_index": frame_index,
                    "index": len(rows),
                    "timestamp": len(rows) / 10.0,
                }
            )
    frame = pd.DataFrame(rows)
    chunk = root / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(chunk / "file-000.parquet", index=False)


def _write_fake_video(root: Path, blob_centers: list[tuple[int, int]], frames_per_episode: int = 5) -> Path:
    """cam_top video: dark background with a green puck blob per episode."""
    video = root / "videos" / "observation.images.cam_top" / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(video), mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 224
    stream.height = 224
    stream.pix_fmt = "yuv420p"
    for cx, cy in blob_centers:
        for _ in range(frames_per_episode):
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame[:] = (40, 40, 45)
            cv2.rectangle(frame, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 200, 90), -1)
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return video


def test_metrics_no_video_returns_error(fake_dataset: Path) -> None:
    _write_fake_parquet(fake_dataset, [5])
    result = experiment_dashboard.read_episode_metrics()
    assert result["ok"] is False
    assert "영상" in result["error"]


def test_metrics_computes_final_dxy_and_passed(fake_dataset: Path) -> None:
    """Episode 0 ends on the goal (PASS); episode 1 stays far away (never passed)."""
    _write_fake_parquet(fake_dataset, [5, 5])
    _write_fake_video(fake_dataset, [(110, 109), (50, 50)])
    result = experiment_dashboard.read_episode_metrics()
    assert result["ok"] is True
    rows = {row["episode_id"]: row for row in result["metrics"]}
    assert len(rows) == 2

    ep0 = rows[0]
    assert ep0["passed"] is True
    assert ep0["final_dxy_cm"] == pytest.approx(0.12, abs=0.4)  # blob centroid at (110,109)
    assert ep0["min_dxy_cm"] == pytest.approx(0.12, abs=0.4)
    assert ep0["last_pass_frame"] == 4

    ep1 = rows[1]
    assert ep1["passed"] is False
    assert ep1["final_dxy_cm"] > 20.0  # blob at (50,50) is far from the goal
    assert ep1["last_pass_frame"] is None


def test_metrics_passed_tracks_in_zone_frames(fake_dataset: Path) -> None:
    """Pin that an in-zone pass latches while final dxy reports the drift end.

    The episode enters the 1cm zone mid-recording, then moves away before
    stopping; it is still 'passed' but its final dxy is large.
    """
    _write_fake_parquet(fake_dataset, [12])
    # blob starts far, moves onto the goal, then moves away for the final frames
    _write_fake_video(fake_dataset, [(60, 60), (110, 109), (120, 120)], frames_per_episode=4)
    result = experiment_dashboard.read_episode_metrics()
    assert result["ok"] is True
    row = result["metrics"][0]
    assert row["passed"] is True
    assert row["last_pass_frame"] == 7  # last frame within the 1cm zone
    assert row["min_dxy_cm"] < 1.0
    assert row["final_dxy_cm"] > 1.0  # ended away from the goal


def test_metrics_skips_mid_write_chunk(fake_dataset: Path) -> None:
    """Cover the finalized prefix while the next chunk is mid-write.

    During collection the active chunk's parquet/video are unreadable; metrics
    must still cover the completed prefix and report a note.
    """
    _write_fake_parquet(fake_dataset, [5])
    _write_fake_video(fake_dataset, [(110, 109)])
    # simulate the active (mid-write) chunk: unreadable parquet + video
    chunk1 = fake_dataset / "data" / "chunk-001"
    chunk1.mkdir(parents=True, exist_ok=True)
    (chunk1 / "file-000.parquet").write_bytes(b"mid-write garbage, not parquet")
    video1 = fake_dataset / "videos" / "observation.images.cam_top" / "chunk-001"
    video1.mkdir(parents=True, exist_ok=True)
    (video1 / "file-000.mp4").write_bytes(b"mid-write garbage, not mp4")

    result = experiment_dashboard.read_episode_metrics()
    assert result["ok"] is True
    assert [row["episode_id"] for row in result["metrics"]] == [0]
    assert result["metrics"][0]["passed"] is True
    assert result["note"]


def test_metrics_skips_mid_write_file_in_same_chunk(fake_dataset: Path) -> None:
    """A mid-write file-001 must not shadow readable file-000 in the chunk.

    LeRobot rolls files within one chunk dir; the finalized prefix must still
    be measured while the newer file in the same dir is being written.
    """
    _write_fake_parquet(fake_dataset, [5])
    _write_fake_video(fake_dataset, [(110, 109)])
    (fake_dataset / "data" / "chunk-000" / "file-001.parquet").write_bytes(b"garbage")
    (fake_dataset / "videos" / "observation.images.cam_top" / "chunk-000" / "file-001.mp4").write_bytes(
        b"garbage"
    )

    result = experiment_dashboard.read_episode_metrics()
    assert result["ok"] is True
    assert [row["episode_id"] for row in result["metrics"]] == [0]
    assert result["metrics"][0]["passed"] is True
    assert result["note"]


def test_metrics_cache_reuses_result_until_dataset_changes(fake_dataset: Path) -> None:
    _write_fake_parquet(fake_dataset, [5])
    video = _write_fake_video(fake_dataset, [(110, 109)])
    first = experiment_dashboard.read_episode_metrics()
    second = experiment_dashboard.read_episode_metrics()
    assert first["metrics"] == second["metrics"]  # cached: no recompute

    # Rewrite the video with a different puck position; the fingerprint change
    # must invalidate the cache and force a recompute.
    os.utime(video, (video.stat().st_atime, video.stat().st_mtime + 5))
    _write_fake_video(fake_dataset, [(150, 150)])
    third = experiment_dashboard.read_episode_metrics()
    assert third["ok"] is True
    assert third["metrics"] != first["metrics"]
    assert third["metrics"][0]["final_dxy_cm"] > 10.0
