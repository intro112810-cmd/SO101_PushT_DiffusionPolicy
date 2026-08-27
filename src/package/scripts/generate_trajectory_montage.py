"""Render one approved model's paper-style fixed-state 40-step trajectory panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate_feedback_artifacts import (
    Capture,
    CaptureOptions,
    capture_rollout,
    load_policy,
)
from so101_pusht_benchmark.evaluation.paper_figure import (
    classify_route_behavior,
    route_mode,
    t_shape_polygons,
    time_gradient,
    trajectory_segments,
    two_mode_score,
)
from so101_pusht_benchmark.evaluation.professor_artifacts import (
    MODEL_ORDER,
    figure_filename,
    get_model_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=MODEL_ORDER)
    parser.add_argument("--artifact")
    parser.add_argument("--environment-seed", required=True, type=int)
    parser.add_argument("--samples", default=12, type=int)
    parser.add_argument("--steps", default=40, type=int)
    parser.add_argument(
        "--behavior",
        choices=("auto", "multimodal", "single-mode", "uncommitted"),
        default="auto",
    )
    parser.add_argument(
        "--output-name",
    )
    return parser.parse_args()


def _fixed_length(points: list[np.ndarray], length: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if len(values) >= length:
        return values[:length]
    return np.concatenate(
        (values, np.repeat(values[-1][None], length - len(values), axis=0))
    )


def plot_paper_panel(
    captures: tuple[Capture, ...],
    output: Path,
    *,
    steps: int,
    label: str,
) -> tuple[float, np.ndarray]:
    trajectories = np.stack(
        [_fixed_length(capture.end_effectors, steps + 1) for capture in captures]
    )
    score, labels = two_mode_score(trajectories)
    figure, axis = plt.subplots(figsize=(6.2, 6.7))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    goal_polygons = list(
        t_shape_polygons(captures[0].goal, yaw=captures[0].goal_yaw)
    )
    block_polygons = list(
        t_shape_polygons(captures[0].blocks[0], yaw=captures[0].block_yaw)
    )
    for polygons, facecolor, edgecolor, alpha, zorder in (
        (goal_polygons, "#98f08d", "#78d96f", 0.85, 0),
        (block_polygons, "#9aabb5", "#718590", 1.0, 2),
    ):
        for vertices in polygons:
            axis.add_patch(
                Polygon(
                    vertices,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.8,
                    alpha=alpha,
                    zorder=zorder,
                )
            )
    colors = time_gradient(steps + 1)
    for trajectory in trajectories:
        collection = LineCollection(
            trajectory_segments(trajectory),
            colors=colors,
            linewidths=2.2,
            alpha=0.88,
            zorder=3,
        )
        collection.set_clip_path(axis.patch)
        axis.add_collection(collection)
    start = trajectories[0, 0]
    axis.add_patch(
        Circle(
            start,
            radius=0.008,
            facecolor="#4389e8",
            edgecolor="#2462b6",
            linewidth=2.2,
            alpha=0.45,
            zorder=4,
        )
    )
    axis.set_xlim(0.04, 0.36)
    axis.set_ylim(-0.16, 0.16)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#c8c8c8")
        spine.set_linewidth(2.0)
    figure.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.10)
    figure.text(
        0.5,
        0.035,
        label,
        ha="center",
        va="center",
        color="#111111",
        fontsize=18,
        weight="bold",
    )
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    return score, labels


def main() -> int:
    args = parse_args()
    root = args.artifact_root.resolve()
    assets = args.asset_dir.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    spec = get_model_spec(args.model)
    policy, identity = load_policy(root, args.artifact or spec.artifact_id, spec.model)
    policy_seeds = tuple(
        args.environment_seed * 1000 + sample for sample in range(args.samples)
    )
    captures = tuple(
        capture_rollout(
            policy,
            identity,
            args.environment_seed,
            f"policy sample {sample}",
            CaptureOptions(
                render=False,
                policy_seed=policy_seed,
                max_steps=args.steps,
            ),
        )
        for sample, policy_seed in enumerate(policy_seeds)
    )
    output = assets / (args.output_name or figure_filename(spec.model))
    trajectories = np.stack(
        [_fixed_length(capture.end_effectors, args.steps + 1) for capture in captures]
    )
    route_modes = [
        route_mode(trajectory, captures[0].blocks[0])
        for trajectory in trajectories
    ]
    opposite_counts = [
        int(np.count_nonzero(np.asarray(route_modes) == mode))
        for mode in (-1, 1)
    ]
    observed_behavior = classify_route_behavior(
        route_modes,
        sample_count=args.samples,
    )
    if args.behavior not in ("auto", observed_behavior):
        raise RuntimeError(
            f"expected {args.behavior}, observed {observed_behavior}: "
            f"{opposite_counts}"
        )
    label = f"{spec.label}: {observed_behavior}"
    score, labels = plot_paper_panel(
        captures,
        output,
        steps=args.steps,
        label=label,
    )
    receipt = {
        "schema": 1,
        "figure_contract": "diffusion-policy-figure-3-fixed-state-v1",
        "model": identity.model,
        "optimizer_updates": identity.optimizer_updates,
        "environment_seed": args.environment_seed,
        "policy_seeds": list(policy_seeds),
        "steps_per_trajectory": args.steps,
        "observed_behavior": observed_behavior,
        "trajectory_quantity": "Fixed_Jaw actual world XY",
        "block_xy": captures[0].blocks[0].tolist(),
        "block_yaw": captures[0].block_yaw,
        "goal_xy": captures[0].goal.tolist(),
        "goal_yaw": captures[0].goal_yaw,
        "end_effector_start_xy": captures[0].end_effectors[0].tolist(),
        "two_mode_score": score,
        "cluster_sizes": [
            int(np.count_nonzero(labels == index)) for index in (0, 1)
        ],
        "route_modes": route_modes,
        "opposite_route_counts": opposite_counts,
        "output": str(output),
    }
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"published {output}", flush=True)
    print(f"published {receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
