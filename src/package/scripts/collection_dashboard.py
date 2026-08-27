#!/usr/bin/env python3
"""Read-only collection progress dashboard for PushT SO-100 datasets.

Shows how many episodes have been recorded by `collect-native --launch`
into a LeRobot dataset directory: episode count, total frames, latest
save time, progress toward the target, and (with --watch) a live
refreshing view that announces each newly saved episode.

Usage:
  python3 -B scripts/collection_dashboard.py --dataset-root <path> [--target 200] [--watch]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_TARGET = 200


def read_status(dataset_root: Path, target: int) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return {
            "ok": False,
            "dataset": str(dataset_root),
            "error": "no meta/info.json yet (dataset is created on first saved episode)",
        }
    info = json.loads(info_path.read_text())
    episodes = int(info.get("total_episodes", 0))
    frames = int(info.get("total_frames", 0))
    fps = info.get("fps")

    latest = None
    for pattern in (
        "data/file-*.parquet",
        "data/chunk-*/file-*.parquet",
        "videos/*/chunk-*/file-*.mp4",
    ):
        for p in dataset_root.glob(pattern):
            m = p.stat().st_mtime
            latest = m if latest is None else max(latest, m)

    now = time.time()
    return {
        "ok": True,
        "dataset": str(dataset_root),
        "episodes": episodes,
        "target": target,
        "remaining": max(0, target - episodes),
        "progress": round(100.0 * episodes / target, 1) if target else 0.0,
        "frames": frames,
        "fps": fps,
        "last_saved": (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest)) if latest else None
        ),
        "seconds_since_save": int(now - latest) if latest else None,
    }


def render(status: dict) -> None:
    if not status["ok"]:
        print(f"[dashboard] {status['error']}")
        return
    bar_len = 20
    filled = int(bar_len * status["episodes"] / status["target"]) if status["target"] else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    print("=== PushT SO-100 Collection Dashboard ===")
    print(f"dataset   : {status['dataset']}")
    print(f"episodes  : {status['episodes']} / {status['target']}  ({status['progress']}%)  [{bar}]")
    print(f"remaining : {status['remaining']}")
    print(f"frames    : {status['frames']}" + (f"  (fps={status['fps']})" if status["fps"] else ""))
    if status["last_saved"]:
        ago = status["seconds_since_save"]
        ago_str = "just now" if ago is not None and ago < 5 else f"{ago}s ago"
        print(f"last save : {status['last_saved']}  ({ago_str})")
    else:
        print("last save : (none yet)")
    print("=" * 42)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path, help="LeRobot dataset dir written by collect-native")
    p.add_argument("--target", type=int, default=DEFAULT_TARGET, help=f"target episode count (default {DEFAULT_TARGET})")
    p.add_argument("--watch", action="store_true", help="refresh every --interval seconds and announce new episodes")
    p.add_argument("--interval", type=float, default=2.0, help="watch refresh interval in seconds")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    last_episodes = -1
    try:
        while True:
            status = read_status(args.dataset_root, args.target)
            if args.watch:
                print("\033[2J\033[H", end="", flush=True)
            render(status)
            if status["ok"] and args.watch:
                if last_episodes >= 0 and status["episodes"] != last_episodes:
                    delta = status["episodes"] - last_episodes
                    if delta > 0:
                        print(f"  >>> NEW EPISODE SAVED: {last_episodes} -> {status['episodes']} (+{delta})")
                    else:
                        print(f"  >>> episodes changed: {last_episodes} -> {status['episodes']} (dataset reset?)")
                last_episodes = status["episodes"]
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[dashboard stopped]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
