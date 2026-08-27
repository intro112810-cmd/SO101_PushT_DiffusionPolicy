#!/usr/bin/env python3
"""Episode deletion utilities for LeRobot-format PushT datasets.

Rewrites a dataset without one episode: backs up the original, drops the
episode's frames from parquet + videos (h264 re-encode), remaps remaining
episode indices, and refreshes meta/info.json + meta/stats.json. The
rewritten dataset stays LeRobot-readable (verified via import-native).

Read-only helpers: list_episodes() only reads the parquet.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 10
_FFMPEG = None


def ffmpeg() -> str:
    global _FFMPEG
    if _FFMPEG is None:
        import imageio_ffmpeg

        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG


def _read_parquet(dataset_root: Path) -> pd.DataFrame:
    files = sorted((dataset_root / "data").rglob("file-*.parquet"))
    if not files:
        raise ValueError(f"no parquet files under {dataset_root / 'data'}")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def list_episodes(dataset_root: Path) -> list[dict]:
    """Per-episode info: id, frames, global index range, time range."""
    dataset_root = Path(dataset_root)
    df = _read_parquet(dataset_root)
    episodes: list[dict] = []
    for ep, sub in df.groupby("episode_index", sort=True):
        episodes.append(
            {
                "episode_id": int(ep),
                "frames": int(len(sub)),
                "index_start": int(sub["index"].min()),
                "index_end": int(sub["index"].max()),
                "ts_start": float(sub["timestamp"].min()),
                "ts_end": float(sub["timestamp"].max()),
            }
        )
    return episodes


def backup_dataset(dataset_root: Path, backup_root: Path) -> Path:
    """Copy the whole dataset (or the parquet/video/meta essentials) to backup."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = backup_root / f"{dataset_root.name}-{stamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dataset_root, destination, dirs_exist_ok=False)
    return destination


def _rewrite_parquet(dataset_root: Path, removed: set[int]) -> pd.DataFrame:
    files = sorted((dataset_root / "data").rglob("file-*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    before = len(df)
    df = df[~df["episode_index"].isin(removed)].copy()
    # remap remaining episodes to 0..N-1 preserving order
    keep = sorted(int(e) for e in df["episode_index"].unique())
    mapping = {old: new for new, old in enumerate(keep)}
    df["episode_index"] = df["episode_index"].map(mapping)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    df["index"] = np.arange(len(df), dtype=np.int64)
    df["frame_index"] = df.groupby("episode_index").cumcount().astype(np.int64)
    df["timestamp"] = (df["frame_index"].to_numpy(dtype=np.float64) / FPS).astype(np.float32)
    # drop any per-frame columns the pipeline does not need; keep canonical set
    keep_cols = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]
    df = df[keep_cols]
    # write preserving the original parquet schema (list<float32> state/action etc.)
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(files[0])
    arrays: list[pa.Array] = []
    for name in schema.names:
        if name not in df.columns:
            raise ValueError(f"column {name} missing from rewritten frame table")
        col = df[name]
        if name in ("observation.state", "action"):
            values = [np.asarray(v, dtype=np.float32) for v in col]
            arrays.append(pa.array(values, type=schema.field(name).type))
        else:
            arrays.append(pa.array(col.to_numpy(), type=schema.field(name).type))
    table = pa.Table.from_arrays(arrays, names=schema.names)
    target = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    for stale in files:
        if stale.resolve() != target.resolve():
            stale.unlink(missing_ok=True)
    pq.write_table(table, target)
    return df


def _rewrite_video(video_path: Path, drop_ranges: list[tuple[int, int]]) -> Path:
    """Re-encode video without [start, end] inclusive frame ranges (h264)."""
    if not drop_ranges:
        return video_path
    conds = [f"not(between(n,{start},{end}))" for start, end in drop_ranges]
    select = "+".join(conds) if len(conds) > 1 else conds[0]
    tmp = video_path.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg(), "-y", "-i", str(video_path),
        "-vf", f"select='{select}',setpts=N/FRAME_RATE/TB",
        "-r", str(FPS), "-c:v", "libx264", "-crf", "30", "-pix_fmt", "yuv420p",
        "-an", str(tmp),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    tmp.replace(video_path)
    return video_path


def _refresh_info(dataset_root: Path, episodes: int, frames: int, codec: str) -> None:
    path = dataset_root / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["total_episodes"] = episodes
    info["total_frames"] = frames
    info["splits"] = {"train": "0:1"}
    for feature in ("observation.images.cam_top", "observation.images.cam_side"):
        info["features"][feature]["info"]["video.codec"] = codec
        info["features"][feature]["info"]["video.fps"] = FPS
    path.write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def _refresh_stats(dataset_root: Path, df: pd.DataFrame) -> None:
    """Recompute numeric stats from the rewritten parquet; keep image stats."""
    path = dataset_root / "meta" / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    numeric_cols = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]
    for col in numeric_cols:
        if col not in df or col not in stats:
            continue
        raw = df[col].to_numpy()
        arr = np.stack([np.asarray(x, dtype=np.float64).reshape(-1) for x in raw])
        quantiles = np.percentile(arr, [1, 10, 50, 90], axis=0)
        stats[col] = {
            "min": [float(v) for v in arr.min(axis=0)],
            "max": [float(v) for v in arr.max(axis=0)],
            "mean": [float(v) for v in arr.mean(axis=0)],
            "std": [float(v) for v in arr.std(axis=0)],
            "count": [int(arr.shape[0])],
            "q01": [float(v) for v in quantiles[0]],
            "q10": [float(v) for v in quantiles[1]],
            "q50": [float(v) for v in quantiles[2]],
            "q90": [float(v) for v in quantiles[3]],
        }
    path.write_text(json.dumps(stats, ensure_ascii=False) + "\n", encoding="utf-8")


def _refresh_episodes_table(dataset_root: Path, df: pd.DataFrame, removed: set[int]) -> None:
    """Rewrite meta/episodes/... episode table to match the rewritten parquet."""
    ep_dir = dataset_root / "meta" / "episodes" / "chunk-000"
    files = sorted(ep_dir.glob("file-*.parquet"))
    if not files:
        return
    table = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    table = table[~table["episode_index"].isin(removed)].copy()
    keep = sorted(int(e) for e in table["episode_index"].unique())
    mapping = {old: new for new, old in enumerate(keep)}
    table["episode_index"] = table["episode_index"].map(mapping)
    new_lengths = df.groupby("episode_index").size()
    table["length"] = table["episode_index"].map(new_lengths).astype(np.int64)
    table["data/chunk_index"] = 0
    table["data/file_index"] = 0
    for cam in ("observation.images.cam_top", "observation.images.cam_side"):
        table[f"videos/{cam}/chunk_index"] = 0
        table[f"videos/{cam}/file_index"] = 0
        table[f"videos/{cam}/from_timestamp"] = 0.0
        table[f"videos/{cam}/to_timestamp"] = (table["length"].to_numpy(dtype=np.float64) / FPS)
    ends = new_lengths.cumsum()
    starts = ends - new_lengths
    table["dataset_from_index"] = table["episode_index"].map(starts).astype(np.int64)
    table["dataset_to_index"] = table["episode_index"].map(ends).astype(np.int64)
    for f in files:
        f.unlink(missing_ok=True)
    table.to_parquet(ep_dir / "file-000.parquet", index=False)


def rewrite_without_episode(dataset_root: Path, episode_id: int, backup_root: Path | None) -> dict:
    dataset_root = Path(dataset_root)
    """Back up, remove one episode, rewrite parquet/videos/meta. Returns report."""
    dataset_root = dataset_root.resolve()
    episodes_before = list_episodes(dataset_root)
    ids_before = {e["episode_id"] for e in episodes_before}
    if episode_id not in ids_before:
        raise ValueError(f"episode {episode_id} not in dataset {dataset_root}")
    backup = None
    if backup_root is not None:
        backup = backup_dataset(dataset_root, Path(backup_root).resolve())

    df = _rewrite_parquet(dataset_root, {episode_id})
    # video frame ranges (global index order == video order)
    target = next(e for e in episodes_before if e["episode_id"] == episode_id)
    drop_range = (target["index_start"], target["index_end"])
    for cam in ("observation.images.cam_top", "observation.images.cam_side"):
        for video in (dataset_root / "videos" / cam).rglob("*.mp4"):
            _rewrite_video(video, [drop_range])
    episodes_after = int(df["episode_index"].nunique())
    frames_after = int(len(df))
    _refresh_info(dataset_root, episodes_after, frames_after, "h264")
    _refresh_episodes_table(dataset_root, df, {episode_id})
    _refresh_stats(dataset_root, df)
    return {
        "deleted_episode": episode_id,
        "episodes_before": len(episodes_before),
        "episodes_after": episodes_after,
        "frames_before": int(sum(e["frames"] for e in episodes_before)),
        "frames_after": frames_after,
        "backup": str(backup) if backup else None,
    }
