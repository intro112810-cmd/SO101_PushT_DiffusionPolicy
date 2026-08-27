"""Measure clockwise/counter-clockwise approach bias in the trained 200ep dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import zarr

from so101_pusht_benchmark.evaluation.frozen_env import load_frozen_pusht
from so101_pusht_benchmark.evaluation.paper_figure import route_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-samples", default=32, type=int)
    parser.add_argument("--route-steps", default=80, type=int)
    return parser.parse_args()


def green_centroid(image: np.ndarray) -> np.ndarray | None:
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    mask = (green > 80) & (green > red + 25) & (green > blue + 20)
    rows, columns = np.nonzero(mask)
    if len(rows) < 8:
        return None
    return np.asarray([columns.mean(), rows.mean()], dtype=np.float64)


def calibrate_pixel_to_world(sample_count: int) -> tuple[np.ndarray, float]:
    environment = load_frozen_pusht(max_steps=1)
    raw = environment.raw_environment
    block_body = raw.model.body("T_block").id
    pixels: list[np.ndarray] = []
    worlds: list[np.ndarray] = []
    try:
        for seed in range(200000, 200000 + sample_count):
            observation, _ = environment.reset(seed=seed)
            image = cv2.resize(np.asarray(observation["cam_top"]), (96, 96))
            centroid = green_centroid(image)
            if centroid is None:
                continue
            pixels.append(np.append(centroid, 1.0))
            worlds.append(raw.data.xpos[block_body, :2].copy())
    finally:
        environment.close()
    design = np.asarray(pixels)
    targets = np.asarray(worlds)
    transform, *_ = np.linalg.lstsq(design, targets, rcond=None)
    residual = design @ transform - targets
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    return transform, rmse


def main() -> int:
    args = parse_args()
    transform, calibration_rmse = calibrate_pixel_to_world(
        args.calibration_samples
    )
    root = zarr.open_group(str(args.dataset.resolve()), mode="r")
    actions = root["data/action"]
    images = root["data/cam_top"]
    episode_ends = np.asarray(root["episode_ends"], dtype=np.int64)
    start = 0
    fixed_jaw_start = np.asarray([0.14803827465088415, 0.005618613878758617])
    modes: list[int] = []
    skipped = 0
    for end in episode_ends:
        centroid = green_centroid(np.asarray(images[start]))
        if centroid is None:
            skipped += 1
            start = int(end)
            continue
        block_center = np.append(centroid, 1.0) @ transform
        episode_actions = np.asarray(
            actions[start : min(int(end), start + args.route_steps)],
            dtype=np.float64,
        )
        trajectory = np.concatenate((fixed_jaw_start[None], episode_actions))
        modes.append(route_mode(trajectory, block_center))
        start = int(end)
    mode_values = np.asarray(modes)
    counts = {
        "clockwise": int(np.count_nonzero(mode_values == -1)),
        "direct_or_uncommitted": int(np.count_nonzero(mode_values == 0)),
        "counter_clockwise": int(np.count_nonzero(mode_values == 1)),
    }
    committed = counts["clockwise"] + counts["counter_clockwise"]
    dominant_share = (
        max(counts["clockwise"], counts["counter_clockwise"]) / committed
        if committed
        else 0.0
    )
    receipt = {
        "schema": 1,
        "dataset": str(args.dataset.resolve()),
        "episode_count": len(episode_ends),
        "analyzed_episode_count": len(modes),
        "segmentation_skipped": skipped,
        "route_steps": args.route_steps,
        "pixel_to_world_calibration_rmse_m": calibration_rmse,
        "route_counts": counts,
        "dominant_committed_route_share": dominant_share,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    print(f"published {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
