"""Render initial states suited to a Figure-3 reverse-then-straight push."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np

from so101_pusht_benchmark.evaluation.frozen_env import load_frozen_pusht
from so101_pusht_benchmark.evaluation.paper_figure import t_shape_polygons


DEFAULT_SEEDS = (100018, 100038, 100061, 100082, 100002, 100026, 100046)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def _yaw(matrix: np.ndarray) -> float:
    return float(np.arctan2(matrix[3], matrix[0]))


def _angle_error(first: float, second: float) -> float:
    return abs(float(np.arctan2(np.sin(first - second), np.cos(first - second))))


def _draw_t(
    axis: plt.Axes,
    center: np.ndarray,
    yaw: float,
    style: tuple[str, str, float],
) -> None:
    facecolor, edgecolor, alpha = style
    for vertices in t_shape_polygons(center, yaw=yaw):
        axis.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=2.0,
                alpha=alpha,
            )
        )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = load_frozen_pusht(max_steps=1)
    raw = env.raw_environment
    block_body = raw.model.body("T_block").id
    goal_body = raw.model.body("T_sign").id
    end_effector_body = raw.model.body("Fixed_Jaw").id
    records: list[dict[str, object]] = []
    try:
        for seed in args.seeds:
            observation, _ = env.reset(seed=seed)
            top = np.asarray(observation["_cam_top_hd"]).copy()
            block = raw.data.xpos[block_body, :2].copy()
            goal = raw.data.xpos[goal_body, :2].copy()
            end_effector = raw.data.xpos[end_effector_body, :2].copy()
            block_yaw = _yaw(raw.data.xmat[block_body])
            goal_yaw = _yaw(raw.data.xmat[goal_body])
            push = goal - block
            push_distance = float(np.linalg.norm(push))
            push_direction = push / push_distance
            front_offset = float(np.dot(end_effector - block, push_direction))
            lateral_offset = float(
                abs(
                    push_direction[0] * (end_effector - block)[1]
                    - push_direction[1] * (end_effector - block)[0]
                )
            )
            yaw_error = _angle_error(block_yaw, goal_yaw)

            figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
            axes[0].imshow(top)
            axes[0].set_title(f"Actual initial camera | seed {seed}", weight="bold")
            axes[0].axis("off")

            axis = axes[1]
            _draw_t(
                axis,
                goal,
                goal_yaw,
                ("#a4ee96", "#6bd66b", 0.65),
            )
            _draw_t(
                axis,
                block,
                block_yaw,
                ("#9dabb4", "#6f8490", 0.95),
            )
            axis.add_patch(
                Circle(
                    end_effector,
                    0.008,
                    facecolor="#8eb8ef",
                    edgecolor="#4f7fc2",
                    linewidth=2.0,
                )
            )
            arrow_length = min(0.045, max(0.025, push_distance))
            axis.annotate(
                "",
                xy=block + push_direction * arrow_length,
                xytext=block,
                arrowprops={"arrowstyle": "->", "color": "#2d9d50", "lw": 3},
            )
            axis.annotate(
                "",
                xy=end_effector - push_direction * 0.035,
                xytext=end_effector,
                arrowprops={"arrowstyle": "->", "color": "#d64c3d", "lw": 3},
            )
            axis.text(
                *(block + push_direction * arrow_length),
                " final push",
                color="#237c3d",
                fontsize=10,
                weight="bold",
            )
            axis.text(
                *(end_effector - push_direction * 0.035),
                " initial reverse",
                color="#aa3329",
                fontsize=10,
                weight="bold",
            )
            points = np.stack((block, goal, end_effector))
            padding = 0.075
            axis.set_xlim(float(points[:, 0].min() - padding), float(points[:, 0].max() + padding))
            axis.set_ylim(float(points[:, 1].min() - padding), float(points[:, 1].max() + padding))
            axis.set_aspect("equal")
            axis.grid(alpha=0.18)
            axis.set_title(
                f"yaw error {np.degrees(yaw_error):.1f}° | "
                f"goal gap {push_distance:.3f} m\n"
                f"arm ahead {front_offset:+.3f} m | lateral {lateral_offset:.3f} m",
                weight="bold",
            )
            figure.suptitle(
                "Candidate: retreat opposite the final push, go around T, then push straight",
                fontsize=14,
                weight="bold",
            )
            figure.tight_layout()
            output = output_dir / f"seed_{seed}_reverse_then_straight.png"
            figure.savefig(output, dpi=160, facecolor="white")
            plt.close(figure)
            records.append(
                {
                    "seed": seed,
                    "yaw_error_deg": float(np.degrees(yaw_error)),
                    "goal_gap_m": push_distance,
                    "arm_ahead_m": front_offset,
                    "lateral_offset_m": lateral_offset,
                    "output": str(output),
                }
            )
    finally:
        env.close()

    images = [cv2.imread(str(Path(record["output"]))) for record in records]
    height = 440
    thumbnails = [
        cv2.resize(image, (960, height), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    rows = [
        np.concatenate(thumbnails[index : index + 2], axis=1)
        if len(thumbnails[index : index + 2]) == 2
        else np.concatenate(
            (thumbnails[index], np.full_like(thumbnails[index], 255)),
            axis=1,
        )
        for index in range(0, len(thumbnails), 2)
    ]
    cv2.imwrite(str(output_dir / "contact_sheet.png"), np.concatenate(rows, axis=0))
    (output_dir / "candidates.json").write_text(
        json.dumps({"criterion": "reverse-then-straight-push", "candidates": records}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"published {len(records)} candidates and contact sheet in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
