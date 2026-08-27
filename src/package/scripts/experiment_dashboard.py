#!/usr/bin/env python3
"""Advanced PushT SO-100 experiment dashboard (research-grade).

Read-only, process-independent live view over the whole benchmark:
  - collection: episodes, frames, per-episode length distribution, save
    event timeline (own state log under ~/.local/state, never the dataset),
    progress toward the target;
  - episodes: per-episode deletion with quarantine backup, plus a
    video-based estimate of the final dxy (puck-to-goal distance at stop,
    recovered from the recorded cam_top video) so failed recordings can be
    identified before deleting;
  - training: 4-model status chain from artifact-index.json (smoke / full /
    bundle / evaluation) plus checkpoint/bundle artifacts;
  - evaluation: anchored four-model comparison report (success rate, steps).

Borrows the file-based live-update model of DVCLive, the run-status table of
Aim/MLflow, and the comparison view of W&B. Never writes into the dataset
root; its own event log lives under the per-user state directory.

Usage:
  python3 -B scripts/experiment_dashboard.py --dataset-root <path> [--target 200] [--port 8890]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE_ROOT = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
BENCH_ROOT = Path(__file__).resolve().parents[1]
ART_ROOT = BENCH_ROOT.parents[1] / "04_experiments/so101_pusht_benchmark"

MODELS = ("dp_cnn", "dp_transformer", "ibc", "lstm_gmm")
MODEL_LABELS = {
    "dp_cnn": "DP-CNN",
    "dp_transformer": "DP-Transformer",
    "ibc": "IBC",
    "lstm_gmm": "LSTM-GMM",
}
FINAL_IDS = {
    "dp_cnn": "dp-cnn-production",
    "dp_transformer": "dp-transformer-production",
    "ibc": "ibc-production",
    "lstm_gmm": "lstm-gmm-production",
}
STAGE_ORDER = (
    "production_smoke_complete_nonfinal",
    "full_training_complete",
    "full_training_bundle_ready",
    "anchored_final_evaluation",
)
STAGE_LABELS = {
    "production_smoke_complete_nonfinal": "smoke (non-final)",
    "full_training_complete": "full training",
    "full_training_bundle_ready": "bundle",
    "anchored_final_evaluation": "evaluated",
}


class ServerState:
    dataset_root: Path
    target: int


SERVER = ServerState()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode_editor import list_episodes, rewrite_without_episode  # noqa: E402


def collector_running(dataset_root: Path | None = None) -> bool:
    """True while the native recorder is alive, optionally for this dataset.

    Without a root, any live recorder blocks deletion (conservative). With a
    root, only a recorder writing that exact dataset blocks it, so cloned
    datasets stay deletable while the user keeps recording the original.
    """
    try:
        import subprocess

        probe = subprocess.run(["pgrep", "-af", "env_human_ee.py"], capture_output=True, text=True)
        if probe.returncode != 0:
            return False
        if dataset_root is None:
            return True
        return str(dataset_root) in probe.stdout
    except Exception:
        return False


def read_episodes() -> dict:
    running = collector_running(SERVER.dataset_root)
    try:
        episodes = list_episodes(SERVER.dataset_root)
        return {"ok": True, "episodes": episodes, "collector_running": running}
    except Exception as exc:
        if running:
            return {
                "ok": False,
                "collector_running": running,
                "error": "수집 중이라 현재 chunk가 쓰기 중입니다 — 녹화 종료(버튼 7) 후 확인하세요.",
            }
        return {"ok": False, "error": str(exc), "collector_running": running}


def delete_episode(episode_id: int, confirm: str) -> dict:
    if collector_running(SERVER.dataset_root):
        return {"ok": False, "code": 409,
                "error": "recorder is running; stop recording before deleting episodes"}
    if confirm != "DELETE":
        return {"ok": False, "code": 400, "error": "confirm token must be exactly DELETE"}
    try:
        report = rewrite_without_episode(
            SERVER.dataset_root, int(episode_id), ART_ROOT / "datasets" / "quarantine"
        )
    except Exception as exc:
        return {"ok": False, "code": 500, "error": str(exc)}
    adjust_flags_after_delete(int(episode_id))
    return {"ok": True, **report}


def _event_log() -> Path:
    root = STATE_ROOT / "so101-pusht-benchmark" / "dashboard"
    root.mkdir(parents=True, exist_ok=True)
    return root / "events.jsonl"


def _record_save_events(current_episodes: int) -> None:
    """Append episode-count increases to the dashboard's own state log."""
    log = _event_log()
    last = None
    if log.exists():
        try:
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    last = json.loads(line).get("episodes")
                except json.JSONDecodeError:
                    continue
        except OSError:
            last = None
    if last is not None and current_episodes > last:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"t": time.time(), "episodes": current_episodes, "delta": current_episodes - last},
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_event_log() -> list[dict]:
    log = _event_log()
    if not log.exists():
        return []
    events: list[dict] = []
    try:
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return events[-200:]


FLAGS_PATH = STATE_ROOT / "so101-pusht-benchmark" / "dashboard" / "episode_flags.json"


def _load_flags() -> dict[str, dict]:
    """Load dashboard-side failed-episode markers (never touches the dataset)."""
    if not FLAGS_PATH.exists():
        return {}
    try:
        raw = json.loads(FLAGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if isinstance(value, dict)}


def _write_flags(flags: dict[str, dict]) -> None:
    FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = FLAGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(flags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(FLAGS_PATH)


def read_flags() -> dict:
    """Return failed-episode markers as a sorted list for the dashboard UI."""
    rows = [
        {"episode_id": int(episode_id), "failed": True, "marked_at": meta.get("marked_at")}
        for episode_id, meta in sorted(_load_flags().items(), key=lambda item: int(item[0]))
    ]
    return {"ok": True, "flags": rows}


def set_failed_flag(episode_id: int | None, failed: bool) -> dict:
    """Mark or unmark an episode as failed; persists to the dashboard state dir."""
    if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id < 0:
        return {"ok": False, "code": 400, "error": "episode_id must be a non-negative int"}
    flags = _load_flags()
    key = str(episode_id)
    if failed:
        flags[key] = {"marked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    else:
        flags.pop(key, None)
    _write_flags(flags)
    return {"ok": True, "flags": read_flags()["flags"]}


def adjust_flags_after_delete(deleted_episode: int) -> None:
    """Shift failed markers to match episode renumbering (0..N-1) after a delete."""
    renumbered: dict[str, dict] = {}
    for raw_id, meta in _load_flags().items():
        episode_id = int(raw_id)
        if episode_id == deleted_episode:
            continue
        new_id = episode_id - 1 if episode_id > deleted_episode else episode_id
        renumbered[str(new_id)] = meta
    _write_flags(renumbered)


# --- Per-episode final success metrics (video-based estimate) ---
#
# The recorder's live "Match dxy" value is never persisted, and the puck's
# start pose is randomized per episode, so the only recoverable per-episode
# success signal is the recorded top-view video. The top camera is fixed,
# looking straight down at the goal's world position (0.25, 0); the green
# puck is detected per frame and its centroid distance to the fixed goal
# centroid is the dxy estimate. This matches the sim's check_xy_pose_match
# within ~0.15 cm whenever the puck yaw is small (successful episodes have
# dyaw <= 5 deg, where the centroid-vs-anchor error is <= 1.3 mm).
#
# Calibration constants are specific to the frozen pushT-so100 scene
# (top_view at (0.25, 0, 0.8), 224x224, default 45 deg fovy).
METRIC_GOAL_CENTROID = (110.4, 108.9)  # px in the 224x224 top view (fixed goal T)
METRIC_PX_PER_M = 338.0  # focal 270.4 px / 0.8 m camera height
METRIC_PASS_M = 0.010  # sim pos_tol from check_xy_pose_match
METRIC_MIN_PIXELS = 30  # ignore tiny masks (noise)

_METRICS_LOCK = threading.Lock()
_METRICS_CACHE: dict[str, object] = {"fingerprint": None, "metrics": None}


def _metrics_fingerprint(dataset_root: Path) -> str | None:
    """Cheap fingerprint of the dataset's video/data files; None if no video."""
    import hashlib

    root = Path(dataset_root)
    videos = sorted(root.glob("videos/*/chunk-*/file-*.mp4"))
    data = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not videos:
        return None
    digest = hashlib.sha256()
    digest.update(str(root.resolve()).encode())
    for path in videos + data:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return digest.hexdigest()


def _chunk_readable(pq_path: Path, vid_path: Path) -> bool:
    """Return True when the chunk parquet and video both open cleanly."""
    import av
    import pandas as pd

    try:
        pd.read_parquet(pq_path)
        container = av.open(str(vid_path))
        try:
            return bool(container.streams.video)
        finally:
            container.close()
    except Exception:
        return False


def _readable_chunk_prefix(root: Path) -> tuple[list[tuple[Path, Path]], int]:
    """Longest prefix of finalized (parquet, video) chunks, plus skipped count.

    LeRobot rewrites the active chunk's parquet/video on every episode save,
    so the current chunk is unreadable while the collector runs. Metrics are
    computed only over the finalized prefix; the trailing count is the number
    of skipped (mid-write or corrupted) chunks.
    """
    parquet_files: dict[tuple[str, str], Path] = {}
    video_files: dict[tuple[str, str], Path] = {}
    for path in sorted((root / "data").rglob("file-*.parquet")):
        parquet_files[(path.parent.name, path.stem)] = path
    for path in sorted((root / "videos" / "observation.images.cam_top").rglob("*.mp4")):
        video_files[(path.parent.name, path.stem)] = path
    pairs: list[tuple[Path, Path]] = []
    for key in sorted(set(parquet_files) & set(video_files)):
        pq_path, vid_path = parquet_files[key], video_files[key]
        if not _chunk_readable(pq_path, vid_path):
            break
        pairs.append((pq_path, vid_path))
    skipped = len(set(parquet_files) | set(video_files)) - len(pairs)
    return pairs, skipped


def _compute_episode_metrics(dataset_root: Path) -> tuple[list[dict], str | None]:
    """Per-episode final/min dxy from the cam_top video (best effort).

    Returns rows aligned with list_episodes order plus a note when the
    dataset has mid-write chunks (collection in progress) that were skipped.
    Requires cv2 + av.
    """
    import cv2
    import av
    import numpy as np
    import pandas as pd

    root = Path(dataset_root)
    pairs, skipped = _readable_chunk_prefix(root)
    if not pairs:
        raise ValueError("no readable parquet/video chunk found")

    lengths: list[int] = []
    for pq_path, _ in pairs:
        frame = pd.read_parquet(pq_path)
        lengths.extend(int(v) for v in frame["episode_index"].value_counts().sort_index().tolist())
    cum = np.cumsum(lengths)
    total_expected = sum(lengths)

    goal = np.array(METRIC_GOAL_CENTROID, dtype=np.float64)
    pass_px = METRIC_PASS_M * METRIC_PX_PER_M

    series: dict[int, list[float]] = {ep: [] for ep in range(len(lengths))}
    fidx = 0
    for _, vid_path in pairs:
        cap = av.open(str(vid_path))
        try:
            vstream = cap.streams.video[0]
            for av_frame in cap.decode(vstream):
                if fidx >= total_expected:
                    break
                img = av_frame.to_ndarray(format="rgb24")
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, (40, 60, 40), (95, 255, 255))
                ys, xs = np.where(mask > 0)
                if len(xs) < METRIC_MIN_PIXELS:
                    fidx += 1
                    continue
                center = np.array((float(xs.mean()), float(ys.mean())), dtype=np.float64)
                ep = int(np.searchsorted(cum, fidx, side="right"))
                if ep < len(lengths):
                    series[ep].append(float(np.linalg.norm(center - goal)))
                fidx += 1
        finally:
            cap.close()

    rows: list[dict] = []
    for ep in range(len(lengths)):
        dxy = np.asarray(series[ep], dtype=np.float64)
        if len(dxy) == 0:
            rows.append(
                {"episode_id": ep, "final_dxy_cm": None, "min_dxy_cm": None,
                 "last_pass_frame": None, "passed": None}
            )
            continue
        below = np.where(dxy <= pass_px)[0]
        rows.append(
            {
                "episode_id": ep,
                "final_dxy_cm": round(float(dxy[-1]) / METRIC_PX_PER_M * 100.0, 2),
                "min_dxy_cm": round(float(dxy.min()) / METRIC_PX_PER_M * 100.0, 2),
                "last_pass_frame": int(below[-1]) if len(below) else None,
                "passed": bool(len(below) > 0),
            }
        )
    note = "수집 중이라 아직 마무리 안 된 에피소드는 제외됨" if skipped else None
    return rows, note


def read_episode_metrics() -> dict:
    """Return cached per-episode dxy metrics, recomputing on dataset change."""
    root = SERVER.dataset_root
    try:
        with _METRICS_LOCK:
            fingerprint = _metrics_fingerprint(root)
            if fingerprint is None:
                return {"ok": False, "error": "cam_top 영상 없음 (아직 저장된 에피소드 없음)"}
            if _METRICS_CACHE["fingerprint"] == fingerprint and _METRICS_CACHE["metrics"] is not None:
                return {
                    "ok": True,
                    "metrics": _METRICS_CACHE["metrics"],
                    "note": _METRICS_CACHE["note"],
                }
            rows, note = _compute_episode_metrics(root)
            _METRICS_CACHE["fingerprint"] = fingerprint
            _METRICS_CACHE["metrics"] = rows
            _METRICS_CACHE["note"] = note
            return {"ok": True, "metrics": rows, "note": note}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def read_collection() -> dict:
    root = SERVER.dataset_root
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "no meta/info.json yet (first saved episode creates it)"}
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ok": False, "error": "meta/info.json unreadable"}
    episodes = int(info.get("total_episodes", 0))
    frames = int(info.get("total_frames", 0))
    fps = info.get("fps")
    target = SERVER.target

    # per-episode lengths from parquet (may be mid-write -> best effort)
    lengths: list[int] = []
    try:
        import pandas as pd

        for p in sorted(root.glob("data/chunk-*/file-*.parquet")):
            df = pd.read_parquet(p)
            lengths.extend(int(v) for v in df["episode_index"].value_counts().sort_index().tolist())
    except Exception:
        lengths = []

    # latest data-file mtimes as a coarse save timeline
    saves: list[float] = []
    for pattern in ("data/chunk-*/file-*.parquet", "videos/*/chunk-*/file-*.mp4"):
        for p in root.glob(pattern):
            saves.append(p.stat().st_mtime)
    saves.sort()

    _record_save_events(episodes)

    return {
        "ok": True,
        "dataset": str(root),
        "episodes": episodes,
        "target": target,
        "remaining": max(0, target - episodes),
        "progress": round(100.0 * episodes / target, 1) if target else 0.0,
        "frames": frames,
        "fps": fps,
        "mean_length": round(sum(lengths) / len(lengths), 1) if lengths else None,
        "length_distribution": lengths,
        "last_saved": (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(saves[-1])) if saves else None
        ),
        "seconds_since_save": int(time.time() - saves[-1]) if saves else None,
        "events": read_event_log(),
    }


def read_experiments() -> dict:
    index_path = ART_ROOT / "artifact-index.json"
    records: dict[str, dict] = {}
    if index_path.exists():
        try:
            artifacts = json.loads(index_path.read_text(encoding="utf-8")).get("artifacts", {})
        except (OSError, json.JSONDecodeError):
            artifacts = {}
        for model in MODELS:
            record = artifacts.get(FINAL_IDS[model], {})
            status = record.get("result_status")
            identity = record.get("identity", {})
            records[model] = {
                "label": MODEL_LABELS[model],
                "artifact_id": FINAL_IDS[model],
                "status": status,
                "status_label": STAGE_LABELS.get(status, status or "not started"),
                "stage_index": STAGE_ORDER.index(status) + 1 if status in STAGE_ORDER else 0,
                "max_stage": len(STAGE_ORDER),
                "policy_class": identity.get("policy_class"),
                "optimizer_updates": identity.get("optimizer_updates"),
                "dataset_digest": (identity.get("dataset_digest") or "")[:12],
            }
    for model in MODELS:
        rec = records.setdefault(
            model,
            {"label": MODEL_LABELS[model], "artifact_id": FINAL_IDS[model], "status": None,
             "status_label": "not started", "stage_index": 0, "max_stage": len(STAGE_ORDER)},
        )
        rec["checkpoint"] = (ART_ROOT / "models" / model / "full" / "checkpoints" / "latest.ckpt").exists()
        rec["bundle"] = (ART_ROOT / "models" / model / "bundle" / "policy.safetensors").exists()
        rec["evaluation"] = (ART_ROOT / "evaluations" / model).exists()
    return {"models": [records[m] for m in MODELS]}


def read_evaluation() -> dict:
    report = ART_ROOT / "reports" / "four-model-final" / "comparison.json"
    if not report.exists():
        return {"ok": False, "error": "no four-model-final comparison yet (after full production)"}
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ok": False, "error": "comparison.json unreadable"}
    rows = []
    for model, metrics in data.get("metrics", {}).items():
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "success_rate": metrics.get("eval/success_rate"),
                "mean_steps": metrics.get("eval/mean_steps"),
                "mean_dxy": metrics.get("eval/mean_dxy"),
                "mean_dyaw": metrics.get("eval/mean_dyaw"),
                "mean_duration_s": metrics.get("eval/mean_duration_s"),
            }
        )
    return {"ok": True, "rows": rows}


MAX_CURVE_POINTS = 500


def _series_stats(values: list[float]) -> dict:
    """Summarize a numeric series (empty-safe) for the dataset-analysis card."""
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None, "n": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(mean, 4),
        "std": round(variance ** 0.5, 4),
        "n": len(values),
    }


def _downsample(points: list[dict], limit: int) -> list[dict]:
    """Keep at most ``limit`` points, preserving the first and last."""
    if limit < 2:
        return points[:limit]
    if len(points) <= limit:
        return points
    stride = (len(points) - 1) / (limit - 1)
    kept = [points[0]]
    kept.extend(points[int(i * stride)] for i in range(1, limit - 1))
    kept.append(points[-1])
    return kept


def read_training_curves() -> dict:
    """Per-model loss/lr curves from the newest LeRobot-style logs.json.txt."""
    curves: dict[str, list[dict]] = {}
    for model in MODELS:
        model_root = ART_ROOT / "models" / model
        logs = sorted(
            model_root.rglob("logs.json.txt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        points: list[dict] = []
        if logs:
            for line in logs[0].read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "global_step" not in row or "train_loss" not in row:
                    continue
                points.append(
                    {
                        "step": int(row["global_step"]),
                        "loss": float(row["train_loss"]),
                        "lr": float(row.get("lr", 0.0)),
                        "epoch": float(row.get("epoch", 0.0)),
                    }
                )
            points = _downsample(points, MAX_CURVE_POINTS)
        curves[model] = points
    return {"ok": True, "curves": curves}


def read_dataset_analysis() -> dict:
    """Per-episode lengths + action XY statistics from a readable parquet."""
    root = SERVER.dataset_root
    try:
        import pandas as pd

        lengths: list[int] = []
        xs: list[float] = []
        ys: list[float] = []
        for parquet in sorted(root.glob("data/chunk-*/file-*.parquet")):
            frame = pd.read_parquet(parquet)
            lengths.extend(
                int(v) for v in frame["episode_index"].value_counts().sort_index().tolist()
            )
            if "action" in frame.columns:
                action_frame = pd.DataFrame(frame["action"].tolist())
                if not action_frame.empty and action_frame.shape[1] >= 2:
                    xs.extend(float(value) for value in action_frame[0].tolist())
                    ys.extend(float(value) for value in action_frame[1].tolist())
    except Exception as exc:
        return {
            "ok": True,
            "readable": False,
            "error": str(exc),
            "note": "parquet 읽기 불가 — 수집 종료 후 표시됩니다",
        }
    return {
        "ok": True,
        "readable": True,
        "lengths": lengths,
        "mean_length": round(sum(lengths) / len(lengths), 1) if lengths else None,
        "action": {"x": _series_stats(xs), "y": _series_stats(ys)},
        "note": None,
    }


PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PushT SO-100 실험 대시보드</title>
<style>
  :root { --bg:#0d1117; --card:#161b26; --border:#2a3346; --text:#e6edf3; --muted:#8b98a9;
          --accent:#4cc38a; --warn:#d29922; --danger:#f85149; --info:#58a6ff; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text);
         margin:0; padding:20px; }
  header { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:14px; }
  h1 { font-size:20px; margin:0; }
  .muted { color: var(--muted); font-size:13px; }
  .tabs { display:flex; gap:6px; margin-bottom:14px; }
  .tab { background:var(--card); border:1px solid var(--border); color:var(--muted); padding:8px 18px;
         border-radius:8px; cursor:pointer; font-size:14px; }
  .tab.active { color:var(--text); border-color:var(--info); }
  .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:12px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px 18px; }
  .card h3 { margin:0 0 10px; font-size:13px; color:var(--muted); font-weight:600; }
  .big { font-size:30px; font-weight:700; }
  .bar { background:var(--border); border-radius:6px; height:12px; overflow:hidden; margin:8px 0; }
  .bar > div { background:var(--accent); height:100%; transition:width .5s; }
  .row { display:flex; justify-content:space-between; margin:6px 0; font-size:14px; }
  .badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; }
  .b-done { background:#123b2a; color:var(--accent); }
  .b-run { background:#3a2c12; color:var(--warn); }
  .b-idle { background:#232b38; color:var(--muted); }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--border); }
  th { color:var(--muted); font-weight:600; }
  .panel { display:none; } .panel.active { display:block; }
  canvas { width:100%; height:180px; display:block; }
  .alert { background:#3a2c12; border:1px solid var(--warn); color:#ffd27a; border-radius:8px;
           padding:8px 12px; margin:8px 0; font-size:13px; }
  input[type=number] { background:var(--card); border:1px solid var(--border); color:var(--text);
           border-radius:6px; padding:6px 10px; font-size:14px; width:150px; }
  button { background:var(--card); border:1px solid var(--border); color:var(--text);
           border-radius:6px; padding:6px 14px; cursor:pointer; font-size:13px; }
  button:hover { border-color:var(--info); }
  .b-fail { background:#3a1d1d; color:#f85149; }
  .nowrap { white-space:nowrap; }
</style>
</head>
<body>
<header>
  <h1>PushT SO-100 실험 대시보드</h1>
  <span class="muted" id="updated">연결 중...</span>
</header>

<div class="tabs">
  <div class="tab active" data-panel="collection">수집</div>
  <div class="tab" data-panel="episodes">에피소드 관리</div>
  <div class="tab" data-panel="training">학습</div>
  <div class="tab" data-panel="evaluation">평가</div>
</div>

<div id="panel-collection" class="panel active">
  <div class="grid" id="c-cards"></div>
  <div class="card" style="margin-top:12px">
    <h3>실패 에피소드 표시</h3>
    <div class="row" style="justify-content:flex-start; gap:8px">
      <input type="number" id="f-id" min="0" placeholder="에피소드 번호">
      <button onclick="markFail()">실패 표시</button>
      <button onclick="clearFail()">표시 해제</button>
    </div>
    <div id="f-list" class="muted" style="margin-top:8px"></div>
  </div>
  <div class="card" style="margin-top:12px">
    <h3>데이터 분석</h3>
    <div id="d-analysis" class="muted"></div>
  </div>
  <div class="card" style="margin-top:12px">
    <h3>에피소드 길이 분포 (프레임)</h3>
    <canvas id="c-dist"></canvas>
  </div>
  <div class="card" style="margin-top:12px">
    <h3>저장 이벤트 타임라인</h3>
    <canvas id="c-timeline"></canvas>
    <div id="c-events"></div>
  </div>
</div>

<div id="panel-episodes" class="panel">
  <div class="card">
    <h3>에피소드 목록</h3>
    <div id="e-warn" class="alert" style="display:none"></div>
    <table id="ep-table"></table>
    <div id="ep-note" class="muted" style="margin-top:8px"></div>
    <div id="ep-result" class="muted" style="margin-top:10px"></div>
  </div>
</div>

<div id="panel-training" class="panel">
  <div class="card">
    <h3>4모델 학습 상태 (artifact-index 기준)</h3>
    <table id="t-table"></table>
  </div>
  <div id="t-curves"></div>
</div>

<div id="panel-evaluation" class="panel">
  <div class="card">
    <h3>성공률 비교</h3>
    <canvas id="e-sr"></canvas>
  </div>
  <div class="card" style="margin-top:12px">
    <h3>평균 스텝 비교</h3>
    <canvas id="e-steps"></canvas>
  </div>
  <div class="card" style="margin-top:12px">
    <h3>4모델 평가 비교</h3>
    <div id="e-note" class="muted" style="margin-bottom:10px"></div>
    <table id="e-table"></table>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const MODELS = ["dp_cnn","dp_transformer","ibc","lstm_gmm"];
const LABELS = {dp_cnn:"DP-CNN",dp_transformer:"DP-Transformer",ibc:"IBC",lstm_gmm:"LSTM-GMM"};
let FAILED_IDS = [];
let EP_METRICS = {};
let EP_METRICS_NOTE = "";

function fmtAgo(sec) {
  if (sec == null) return "-";
  if (sec < 5) return "방금 전";
  if (sec < 60) return sec + "s 전";
  if (sec < 3600) return Math.floor(sec/60) + "m 전";
  return Math.floor(sec/3600) + "h 전";
}

function drawBars(canvas, values, color) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth * dpr, h = canvas.clientHeight * dpr;
  canvas.width = w; canvas.height = h;
  ctx.clearRect(0,0,w,h);
  if (!values || !values.length) {
    ctx.fillStyle = "#8b98a9"; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("데이터 없음", w/2, h/2); return;
  }
  const max = Math.max(...values, 1);
  const n = values.length, gap = 2;
  const bw = Math.max(2, (w / n) - gap);
  for (let i=0;i<n;i++) {
    const bh = Math.max(2, (values[i]/max) * (h-6));
    ctx.fillStyle = color || "#4cc38a";
    ctx.fillRect(i*(bw+gap), h-bh, bw, bh);
  }
}

function drawLine(canvas, ys, label) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth * dpr, h = canvas.clientHeight * dpr;
  canvas.width = w; canvas.height = h;
  ctx.clearRect(0,0,w,h);
  if (!ys || ys.length < 1) {
    ctx.fillStyle = "#8b98a9"; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("아직 저장 이벤트 없음 (첫 episode 저장 시 표시)", w/2, h/2); return;
  }
  const max = Math.max(...ys, 1);
  ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 2; ctx.beginPath();
  ys.forEach((v,i) => {
    const x = (i/(ys.length-1 || 1)) * (w-8) + 4;
    const y = h - 6 - (v/max)*(h-12);
    i ? ctx.lineTo(x,y) : ctx.moveTo(x,y);
  });
  ctx.stroke();
  ctx.fillStyle = "#58a6ff";
  ys.forEach((v,i) => {
    const x = (i/(ys.length-1 || 1)) * (w-8) + 4;
    const y = h - 6 - (v/max)*(h-12);
    ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fill();
  });
  ctx.fillStyle = "#8b98a9"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
  ctx.fillText(label + " (누적 " + ys[ys.length-1] + ")", w/2, h-2);
}

function drawCurve(canvas, pts, label, color) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth * dpr, h = canvas.clientHeight * dpr;
  canvas.width = w; canvas.height = h;
  ctx.clearRect(0,0,w,h);
  if (!pts || !pts.length) {
    ctx.fillStyle = "#8b98a9"; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("아직 학습 곡선 없음", w/2, h/2); return;
  }
  if (pts.length === 1) {
    ctx.fillStyle = color || "#4cc38a";
    ctx.beginPath(); ctx.arc(w/2, h/2, 3, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = "#8b98a9"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(label + " · step " + pts[0].step + " · loss " + pts[0].loss.toFixed(3), w/2, h - 2);
    return;
  }
  const xs = pts.map(p => p.step);
  const ys = pts.map(p => p.loss);
  const minX = xs[0], maxX = xs[xs.length-1];
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeY = (maxY - minY) || 1;
  ctx.strokeStyle = color || "#4cc38a"; ctx.lineWidth = 2; ctx.beginPath();
  pts.forEach((p, i) => {
    const x = (p.step - minX) / (maxX - minX || 1) * (w - 10) + 5;
    const y = h - 6 - ((p.loss - minY) / rangeY) * (h - 18);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#8b98a9"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
  ctx.fillText(label + " · step " + minX + "~" + maxX + " · loss " + minY.toFixed(3) + "~" + maxY.toFixed(3), w/2, h - 2);
}

function drawBarChart(canvas, values, labels, color) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth * dpr, h = canvas.clientHeight * dpr;
  canvas.width = w; canvas.height = h;
  ctx.clearRect(0,0,w,h);
  const data = values.map((v, i) => ({v: v, label: labels[i]})).filter(d => d.v != null);
  if (!data.length) {
    ctx.fillStyle = "#8b98a9"; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("데이터 없음", w/2, h/2); return;
  }
  const max = Math.max(...data.map(d => d.v), 1);
  const n = data.length, gap = 10;
  const bw = Math.max(12, (w - gap * (n + 1)) / n);
  data.forEach((d, i) => {
    const bh = Math.max(2, (d.v / max) * (h - 40));
    const x = gap + i * (bw + gap);
    ctx.fillStyle = color || "#4cc38a";
    ctx.fillRect(x, h - 24 - bh, bw, bh);
    ctx.fillStyle = "#8b98a9"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(d.label, x + bw/2, h - 8);
    ctx.fillStyle = "#e6edf3"; ctx.font = "11px sans-serif";
    ctx.fillText(String(Number(d.v.toFixed(1))), x + bw/2, h - 30 - bh);
  });
}

function renderCollection(c) {
  $("updated").textContent = "갱신: " + new Date().toLocaleTimeString("ko-KR");
  if (!c.ok) { $("c-cards").innerHTML = '<div class="card"><span class="muted">' + c.error + "</span></div>"; return; }
  const pct = Math.min(100, c.progress);
  $("c-cards").innerHTML =
    '<div class="card"><h3>에피소드</h3><div class="big">' + c.episodes + " / " + c.target + '</div>' +
    '<div class="bar"><div style="width:' + pct + '%"></div></div>' +
    '<div class="row"><span>진행률</span><span>' + c.progress + "%</span></div>" +
    '<div class="row"><span>남은 에피소드</span><span>' + c.remaining + "</span></div></div>" +
    '<div class="card"><h3>데이터</h3><div class="big">' + c.frames + '</div>' +
    '<div class="row"><span>프레임 (fps=' + c.fps + ")</span></div>" +
    '<div class="row"><span>에피소드 평균 길이</span><span>' + (c.mean_length ?? "-") + " 프레임</span></div></div>" +
    '<div class="card"><h3>최근 저장</h3><div class="big" style="font-size:18px">' + (c.last_saved || "-") + "</div>" +
    '<div class="row"><span>' + fmtAgo(c.seconds_since_save) + "</span></div></div>" +
    '<div class="card"><h3>데이터셋</h3><div class="muted" style="word-break:break-all">' + c.dataset + "</div></div>";
  drawBars($("c-dist"), c.length_distribution);
  const events = c.events || [];
  drawLine($("c-timeline"), events.map(e => e.episodes), "에피소드");
  const evHtml = events.slice(-6).reverse().map(e =>
    '<div class="alert">>>> 저장됨: ' + e.episodes + ' episode (이벤트 ' + new Date(e.t*1000).toLocaleTimeString("ko-KR") + ")</div>"
  ).join("");
  $("c-events").innerHTML = evHtml;
}

function stageBadge(r) {
  if (r.status === "anchored_final_evaluation") return '<span class="badge b-done">평가 완료</span>';
  if (r.status) return '<span class="badge b-run">' + r.status_label + "</span>";
  return '<span class="badge b-idle">미시작</span>';
}

function stageBar(r) {
  const pct = r.max_stage ? Math.round(100 * r.stage_index / r.max_stage) : 0;
  return '<div class="bar" style="width:160px"><div style="width:' + pct + '%"></div></div>';
}

function renderTrainingCurves(curves) {
  const html = MODELS.map(m =>
    '<div class="card" style="margin-top:12px"><h3>' + LABELS[m] + ' 학습 곡선</h3><canvas id="curve-' + m + '"></canvas></div>'
  ).join("");
  $("t-curves").innerHTML = html;
  MODELS.forEach(m => {
    const el = document.getElementById("curve-" + m);
    if (el) drawCurve(el, (curves && curves[m]) || [], LABELS[m], null);
  });
}

function renderEvaluationCharts(e) {
  const sr = $("e-sr"), steps = $("e-steps");
  if (!e.ok || !e.rows || !e.rows.length) {
    drawBarChart(sr, [], [], "#58a6ff");
    drawBarChart(steps, [], [], "#d29922");
    return;
  }
  drawBarChart(sr, e.rows.map(r => r.success_rate != null ? r.success_rate * 100 : null), e.rows.map(r => r.label), "#58a6ff");
  drawBarChart(steps, e.rows.map(r => r.mean_steps), e.rows.map(r => r.label), "#d29922");
}

function renderDatasetAnalysis(d) {
  const el = $("d-analysis");
  if (!d || !d.ok) { el.textContent = "분석 데이터 없음"; return; }
  if (!d.readable) { el.textContent = d.note || "분석 불가"; return; }
  const x = (d.action && d.action.x) || {}, y = (d.action && d.action.y) || {};
  el.innerHTML =
    '<div class="row"><span>에피소드</span><span>' + (d.lengths ? d.lengths.length : 0) + '개 (평균 ' + (d.mean_length ?? "-") + ' 프레임)</span></div>' +
    '<div class="row"><span>action X</span><span>[' + (x.min ?? "-") + ' ~ ' + (x.max ?? "-") + '] mean ' + (x.mean ?? "-") + ' std ' + (x.std ?? "-") + '</span></div>' +
    '<div class="row"><span>action Y</span><span>[' + (y.min ?? "-") + ' ~ ' + (y.max ?? "-") + '] mean ' + (y.mean ?? "-") + ' std ' + (y.std ?? "-") + '</span></div>' +
    '<div class="row"><span>샘플</span><span>' + (x.n ?? 0) + ' action</span></div>';
  if (d.lengths) drawBars($("c-dist"), d.lengths, "#4cc38a");
}

function renderTraining(t) {
  const rows = t.models.map(r =>
    "<tr><td><b>" + r.label + "</b><div class='muted'>" + r.artifact_id + "</div></td>" +
    "<td>" + stageBadge(r) + "</td>" +
    "<td>" + stageBar(r) + " " + r.stage_index + "/" + r.max_stage + "</td>" +
    "<td>" + (r.policy_class || "-") + "</td>" +
    "<td>" + (r.optimizer_updates ?? "-") + "</td>" +
    "<td>" + (r.checkpoint ? "O" : "-") + "</td>" +
    "<td>" + (r.bundle ? "O" : "-") + "</td>" +
    "<td>" + (r.evaluation ? "O" : "-") + "</td></tr>"
  ).join("");
  $("t-table").innerHTML =
    "<tr><th>모델</th><th>상태</th><th>진행</th><th>정책 클래스</th><th>업데이트</th><th>체크포인트</th><th>번들</th><th>평가</th></tr>" + rows;
}

function renderEvaluation(e) {
  $("e-note").textContent = e.ok ? "" : e.error;
  renderEvaluationCharts(e);
  if (!e.ok) { $("e-table").innerHTML = ""; return; }
  const rows = e.rows.map(r => {
    const sr = r.success_rate != null ? (r.success_rate * 100).toFixed(1) + "%" : "-";
    const pct = r.success_rate != null ? Math.round(r.success_rate * 100) : 0;
    const bar = r.success_rate != null
      ? '<div class="bar" style="width:120px"><div style="width:' + pct + '%"></div></div>'
      : '-';
    return "<tr><td><b>" + r.label + "</b></td>" +
      "<td>" + sr + " " + bar + "</td>" +
      "<td>" + (r.mean_steps ?? "-") + "</td>" +
      "<td>" + (r.mean_dxy != null ? r.mean_dxy.toFixed(4) : "-") + "</td>" +
      "<td>" + (r.mean_dyaw != null ? r.mean_dyaw.toFixed(2) : "-") + "</td></tr>";
  }).join("");
  $("e-table").innerHTML =
    "<tr><th>모델</th><th>성공률</th><th>평균 스텝</th><th>평균 dxy</th><th>평균 dyaw(°)</th></tr>" + rows;
}

function metricBadge(m) {
  if (!m || m.final_dxy_cm == null) return '<span class="muted">-</span>';
  if (m.passed === false) return '<span class="badge b-fail">미도달</span>';
  if (m.final_dxy_cm <= 1.0) return '<span class="badge b-done">PASS</span>';
  if (m.final_dxy_cm <= 2.5) return '<span class="badge b-run">근접</span>';
  return '<span class="badge b-fail">실패 종료</span>';
}

function renderEpisodes(ep) {
  const warn = $("e-warn");
  if (ep.collector_running) {
    warn.style.display = "";
    warn.textContent = "수집기가 실행 중입니다 — 삭제는 수집창을 종료한 뒤에 가능합니다.";
  } else {
    warn.style.display = "none";
  }
  if (!ep.ok) {
    $("ep-table").innerHTML = '<tr><td class="muted">' + (ep.error || "에피소드 정보 없음") + "</td></tr>";
    return;
  }
  const rows = ep.episodes.map(r => {
    const failed = FAILED_IDS.includes(r.episode_id) ? ' <span class="badge b-fail">실패</span>' : "";
    const m = EP_METRICS[r.episode_id];
    const finalDxy = m && m.final_dxy_cm != null ? m.final_dxy_cm.toFixed(2) : "-";
    const minDxy = m && m.min_dxy_cm != null ? m.min_dxy_cm.toFixed(2) : "-";
    const lastPass = m && m.last_pass_frame != null ? m.last_pass_frame + " / " + r.frames : "-";
    return "<tr><td><b>episode " + r.episode_id + "</b>" + failed + "</td>" +
      "<td>" + r.frames + " 프레임</td>" +
      "<td>" + r.ts_start.toFixed(1) + "s ~ " + r.ts_end.toFixed(1) + "s</td>" +
      "<td class='nowrap'>" + finalDxy + " cm</td>" +
      "<td class='nowrap'>" + minDxy + " cm</td>" +
      "<td>" + lastPass + "</td>" +
      "<td>" + metricBadge(m) + "</td>" +
      "<td><button onclick='deleteEp(" + r.episode_id + ")'>삭제</button></td></tr>";
  }).join("");
  $("ep-table").innerHTML =
    "<tr><th>에피소드</th><th>프레임</th><th>시간 범위</th><th>종료 dxy</th><th>최소 dxy</th><th>마지막 PASS</th><th>판정</th><th></th></tr>" + rows;
  $("ep-note").textContent = EP_METRICS_NOTE;
}

async function deleteEp(id) {
  const confirm = prompt("episode " + id + " 삭제. 복원은 quarantine 백업에서 가능합니다.\\n확인하려면 DELETE 입력:", "");
  if (confirm !== "DELETE") { $("ep-result").textContent = "취소됨"; return; }
  const res = await fetch("/api/episodes/delete", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({episode_id: id, confirm: confirm}),
  });
  const out = await res.json();
  $("ep-result").textContent = res.status === 200
    ? "삭제 완료: " + out.episodes_before + " -> " + out.episodes_after + " episode (백업: " + out.backup + ")"
    : "실패 (" + res.status + "): " + (out.error || "");
  refresh();
}

function renderFailed(flags) {
  FAILED_IDS = (flags || []).map(f => f.episode_id);
  $("f-list").textContent = FAILED_IDS.length
    ? "실패 표시됨: " + FAILED_IDS.join(", ") + " (" + FAILED_IDS.length + "개)"
    : "실패 표시된 에피소드 없음";
}

async function postFlag(id, failed) {
  try {
    const res = await fetch("/api/flags", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({episode_id: id, failed: failed}),
    });
    const out = await res.json();
    if (out.ok) { renderFailed(out.flags); }
    else { $("f-list").textContent = "실패: " + (out.error || res.status); }
  } catch (err) { $("f-list").textContent = "실패: " + err; }
}

function markFail() {
  const v = parseInt($("f-id").value, 10);
  if (isNaN(v) || v < 0) { $("f-list").textContent = "올바른 에피소드 번호를 입력하세요"; return; }
  postFlag(v, true);
}

function clearFail() {
  const v = parseInt($("f-id").value, 10);
  if (isNaN(v) || v < 0) { $("f-list").textContent = "올바른 에피소드 번호를 입력하세요"; return; }
  postFlag(v, false);
}

async function refresh() {
  try {
    const [c, ep, t, e, f, tc, da, em] = await Promise.all([
      fetch("/api/collection").then(r => r.json()),
      fetch("/api/episodes").then(r => r.json()),
      fetch("/api/experiments").then(r => r.json()),
      fetch("/api/evaluation").then(r => r.json()),
      fetch("/api/flags").then(r => r.json()),
      fetch("/api/training-curves").then(r => r.json()),
      fetch("/api/dataset-analysis").then(r => r.json()),
      fetch("/api/episode-metrics").then(r => r.json()),
    ]);
    EP_METRICS = {};
    EP_METRICS_NOTE = "";
    if (em && em.ok) {
      (em.metrics || []).forEach(m => { EP_METRICS[m.episode_id] = m; });
      EP_METRICS_NOTE = (em.note ? em.note + " — " : "") +
        "종료 dxy: cam_top 영상에서 초록 퍽 중심 vs 고정 목표 중심 거리 추정치 (±0.15cm, yaw 미포함). " +
        "PASS = 종료 시점 ≤ 1.0cm (sim 기준 0.010m), 미도달 = 전 구간에서 1.0cm 이내 도달 없음. " +
        "기존 녹화는 성공 판정 값이 저장되지 않아 영상 분석으로 추정합니다.";
    } else if (em && em.error) {
      EP_METRICS_NOTE = "종료 dxy 측정 불가: " + em.error;
    }
    renderCollection(c); renderEpisodes(ep); renderTraining(t); renderEvaluation(e); renderFailed(f.flags);
    renderTrainingCurves(tc && tc.curves); renderDatasetAnalysis(da);
  } catch (err) {
    $("updated").textContent = "연결 실패: " + err;
  }
}

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(x => x.classList.remove("active"));
    tab.classList.add("active");
    $("panel-" + tab.dataset.panel).classList.add("active");
  });
});

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path == "/api/episodes/delete":
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            except json.JSONDecodeError:
                payload = {}
            result = delete_episode(payload.get("episode_id"), str(payload.get("confirm", "")))
            code = 200 if result.get("ok") else int(result.get("code", 500))
            body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/flags":
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            except json.JSONDecodeError:
                payload = {}
            result = set_failed_flag(
                payload.get("episode_id"), bool(payload.get("failed", True))
            )
            code = 200 if result.get("ok") else int(result.get("code", 500))
            body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send(404, b"not found", "text/plain")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/episodes":
            self._send(200, json.dumps(read_episodes(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path == "/api/collection":
            self._send(200, json.dumps(read_collection(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path == "/api/flags":
            self._send(200, json.dumps(read_flags(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path == "/api/episode-metrics":
            self._send(200, json.dumps(read_episode_metrics(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path == "/api/training-curves":
            self._send(200, json.dumps(read_training_curves(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path == "/api/dataset-analysis":
            self._send(200, json.dumps(read_dataset_analysis(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path == "/api/experiments":
            self._send(200, json.dumps(read_experiments(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif self.path == "/api/evaluation":
            self._send(200, json.dumps(read_evaluation(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[experiment-dashboard] %s\n" % (fmt % args))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--target", type=int, default=200)
    p.add_argument("--port", type=int, default=8890)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    SERVER.dataset_root = args.dataset_root
    SERVER.target = args.target
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        f"[experiment-dashboard] listening on http://127.0.0.1:{args.port} "
        f"(dataset={args.dataset_root}, target={args.target})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[experiment-dashboard] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
