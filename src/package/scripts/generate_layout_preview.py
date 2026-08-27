#!/usr/bin/env python3
"""Create a 3D rollout layout preview with top-camera and 2D block panels."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import sys

import cv2
import imageio.v2 as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate_feedback_artifacts import Capture, capture_rollout, load_policy


_PANEL_SIZE = 360
_HEADER = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--seed", default=100006, type=int)
    return parser.parse_args()


def label_panel(title: str, image: np.ndarray) -> np.ndarray:
    panel = np.full((_PANEL_SIZE, 380, 3), (14, 24, 38), dtype=np.uint8)
    panel[_HEADER + 8 : _HEADER + 8 + 324, 24 : 24 + 324] = cv2.resize(
        image, (324, 324), interpolation=cv2.INTER_CUBIC
    )
    cv2.putText(panel, title, (18, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (230, 240, 250), 1)
    return panel


def map_point(point: np.ndarray) -> tuple[int, int]:
    x = int((float(point[0]) - 0.05) / 0.40 * 324) + 24
    y = int((0.22 - float(point[1])) / 0.44 * 324) + _HEADER + 8
    return x, y


def draw_block_state(capture: Capture, step: int) -> np.ndarray:
    panel = np.full((_PANEL_SIZE, 380, 3), (14, 24, 38), dtype=np.uint8)
    cv2.putText(
        panel,
        f"2D BLOCK STATE | step {step}",
        (18, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (230, 240, 250),
        1,
    )
    top, left, size = _HEADER + 8, 24, 324
    cv2.rectangle(panel, (left, top), (left + size, top + size), (40, 54, 72), -1)
    for fraction in (0.25, 0.5, 0.75):
        line = left + int(size * fraction)
        cv2.line(panel, (line, top), (line, top + size), (58, 73, 91), 1)
        line = top + int(size * fraction)
        cv2.line(panel, (left, line), (left + size, line), (58, 73, 91), 1)
    target = np.asarray(
        [map_point(point) for point in capture.targets[: step + 1]], dtype=np.int32
    )
    cv2.polylines(panel, [target], False, (80, 220, 180), 2)
    goal = map_point(capture.goal)
    cv2.line(panel, (goal[0] - 36, goal[1] - 24), (goal[0] + 36, goal[1] - 24), (160, 170, 185), 9)
    cv2.line(panel, (goal[0], goal[1] - 24), (goal[0], goal[1] + 28), (160, 170, 185), 9)
    block_start = map_point(capture.blocks[0])
    block_end = map_point(capture.blocks[step])
    cv2.circle(panel, block_start, 7, (200, 130, 255), -1)
    cv2.circle(panel, block_end, 9, (255, 190, 70), -1)
    cv2.putText(panel, "mocap target", (32, 346), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 220, 180), 1)
    return panel


def compose_frame(capture: Capture, step: int) -> np.ndarray:
    canvas = np.full((720, 1280, 3), (8, 16, 28), dtype=np.uint8)
    main = cv2.resize(capture.frames[step], (900, 720), interpolation=cv2.INTER_CUBIC)
    canvas[:, :900] = main
    canvas[:360, 900:] = label_panel(
        "TOP CAMERA (policy input)", capture.top_frames[step]
    )
    canvas[360:, 900:] = draw_block_state(capture, step)
    cv2.rectangle(canvas, (0, 0), (900, 58), (8, 16, 28), -1)
    cv2.putText(canvas, "3D POLICY ROLLOUT", (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 255, 255), 2)
    return canvas


def compose_preview(capture: Capture) -> np.ndarray:
    return compose_frame(capture, len(capture.frames) - 1)


def three_view_frames(capture: Capture) -> Iterable[np.ndarray]:
    title = np.full((720, 1280, 3), (8, 16, 28), dtype=np.uint8)
    cv2.putText(
        title,
        capture.label,
        (110, 315),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.65,
        (255, 255, 255),
        4,
    )
    cv2.putText(
        title,
        f"seed {capture.seed} | synchronized 3-view rollout | 4x playback",
        (110, 385),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (160, 220, 255),
        2,
    )
    for _ in range(20):
        yield title
    for step in range(len(capture.frames)):
        yield compose_frame(capture, step)


def main() -> int:
    args = parse_args()
    assets = args.asset_dir.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    policy, identity = load_policy(
        args.artifact_root.resolve(), "local-dp_transformer-seed0", "dp_transformer"
    )
    capture = capture_rollout(policy, identity, args.seed, "SUCCESS: layout preview")
    if not capture.success:
        raise RuntimeError(f"seed {args.seed} must be successful for the layout preview")
    output = assets / "2026-08-20_dp_transformer_feedback_layout_preview.png"
    iio.imwrite(output, compose_preview(capture))
    print(f"published {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
