"""Rank fixed environment states by two-mode DP-Transformer trajectory separation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_feedback_artifacts import (
    CaptureOptions,
    capture_rollout,
    load_policy,
)
from so101_pusht_benchmark.evaluation.paper_figure import route_mode, two_mode_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed-start", default=100000, type=int)
    parser.add_argument("--seed-count", default=20, type=int)
    parser.add_argument("--samples", default=6, type=int)
    parser.add_argument("--steps", default=40, type=int)
    return parser.parse_args()


def _fixed_length(points: list[np.ndarray], length: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if len(values) >= length:
        return values[:length]
    return np.concatenate(
        (values, np.repeat(values[-1][None], length - len(values), axis=0))
    )


def main() -> int:
    args = parse_args()
    policy, identity = load_policy(
        args.artifact_root.resolve(),
        "local-dp_transformer-seed0",
        "dp_transformer",
    )
    records: list[dict[str, object]] = []
    options = CaptureOptions(render=False, max_steps=args.steps)
    for environment_seed in range(args.seed_start, args.seed_start + args.seed_count):
        captures = tuple(
            capture_rollout(
                policy,
                identity,
                environment_seed,
                f"screen sample {sample}",
                CaptureOptions(
                    render=options.render,
                    policy_seed=environment_seed * 1000 + sample,
                    max_steps=options.max_steps,
                ),
            )
            for sample in range(args.samples)
        )
        trajectories = np.stack(
            [
                _fixed_length(capture.end_effectors, args.steps + 1)
                for capture in captures
            ]
        )
        score, labels = two_mode_score(trajectories)
        modes = [
            route_mode(trajectory, captures[0].blocks[0])
            for trajectory in trajectories
        ]
        opposite_counts = [
            int(np.count_nonzero(np.asarray(modes) == mode)) for mode in (-1, 1)
        ]
        records.append(
            {
                "environment_seed": environment_seed,
                "score": score,
                "cluster_sizes": [
                    int(np.count_nonzero(labels == index)) for index in (0, 1)
                ],
                "route_modes": modes,
                "opposite_route_counts": opposite_counts,
                "verified_opposite_routes": min(opposite_counts) >= 2,
                "block_xy": captures[0].blocks[0].tolist(),
                "block_yaw": captures[0].block_yaw,
                "end_effector_xy": captures[0].end_effectors[0].tolist(),
            }
        )
        print(
            f"seed={environment_seed} score={score:.6f} "
            f"clusters={records[-1]['cluster_sizes']} routes={opposite_counts}",
            flush=True,
        )
    ranked = sorted(
        records,
        key=lambda item: (
            bool(item["verified_opposite_routes"]),
            min(item["opposite_route_counts"]),
            float(item["score"]),
        ),
        reverse=True,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "samples_per_state": args.samples,
                "steps": args.steps,
                "ranked_states": ranked,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"published {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
