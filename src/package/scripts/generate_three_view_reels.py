"""Generate outcome-grouped 3-view reels with a visible final-pose hold."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import cast

import cv2
import imageio.v2 as iio
import numpy as np

from generate_feedback_artifacts import Capture, capture_rollout, load_policy
from generate_layout_preview import compose_frame
from so101_pusht_benchmark.evaluation.reel import (
    RolloutMetric,
    filter_replay_outcomes,
    hold_frame_count,
    repeat_rollouts_to_target,
    select_reel_rollouts,
)
from so101_pusht_benchmark.evaluation.professor_artifacts import (
    MODEL_ORDER,
    get_model_spec,
    reel_filename,
    validate_metrics_receipt,
)


_FPS = 40
_TITLE_FRAMES = 20


def _title_frame(capture: Capture) -> np.ndarray:
    frame = np.full((720, 1280, 3), (8, 16, 28), dtype=np.uint8)
    outcome = "SUCCESS" if capture.success else "FAILURE"
    color = (90, 220, 120) if capture.success else (90, 120, 255)
    cv2.putText(
        frame,
        f"{outcome} ROLLOUT",
        (110, 315),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.65,
        color,
        4,
    )
    cv2.putText(
        frame,
        f"seed {capture.seed} | synchronized 3-view | 4x playback",
        (110, 385),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (220, 230, 245),
        2,
    )
    return frame


def _outcome_hold_frame(capture: Capture, hold_seconds: float) -> np.ndarray:
    frame = compose_frame(capture, len(capture.frames) - 1)
    outcome = "SUCCESS" if capture.success else "FAILURE"
    color = (34, 139, 74) if capture.success else (40, 55, 190)
    cv2.rectangle(frame, (0, 650), (1280, 720), color, -1)
    cv2.putText(
        frame,
        f"{outcome} | final pose hold {hold_seconds:.1f}s",
        (32, 696),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (255, 255, 255),
        3,
    )
    return frame


def write_reel(
    captures: Sequence[Capture],
    output: Path,
    *,
    hold_seconds: float,
    hold_frames: int,
) -> None:
    """Stream one reel to disk without retaining all composed frames."""
    with iio.get_writer(output, fps=_FPS, macro_block_size=1) as writer:
        for capture in captures:
            title = _title_frame(capture)
            for _ in range(_TITLE_FRAMES):
                writer.append_data(title)
            for step in range(len(capture.frames)):
                writer.append_data(compose_frame(capture, step))
            hold = _outcome_hold_frame(capture, hold_seconds)
            for _ in range(hold_frames):
                writer.append_data(hold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=MODEL_ORDER)
    parser.add_argument("--artifact")
    parser.add_argument("--target-seconds", default=120.0, type=float)
    parser.add_argument("--hold-seconds", default=2.0, type=float)
    parser.add_argument(
        "--outcome",
        choices=("both", "success", "failure"),
        default="both",
    )
    parser.add_argument("--skip-outcome-drift", action="store_true")
    parser.add_argument("--repeat-to-target", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.artifact_root.resolve()
    assets = args.asset_dir.resolve()
    metrics = cast(dict[str, object], json.loads(args.metrics.read_text()))
    validate_metrics_receipt(metrics, expected_model=args.model)
    rollouts = cast(list[RolloutMetric], metrics["rollouts"])
    hold_frames = hold_frame_count(seconds=args.hold_seconds, fps=_FPS)
    target_frames = round(args.target_seconds * _FPS)
    spec = get_model_spec(args.model)
    policy, identity = load_policy(root, args.artifact or spec.artifact_id, spec.model)
    outcomes = {
        "both": (True, False),
        "success": (True,),
        "failure": (False,),
    }[args.outcome]
    for success in outcomes:
        selected = select_reel_rollouts(
            rollouts,
            success=success,
            target_frames=target_frames,
            title_frames=_TITLE_FRAMES,
            hold_frames=hold_frames,
        )
        captures = tuple(
            capture_rollout(
                policy,
                identity,
                rollout["seed"],
                f"{spec.label} | {'SUCCESS' if success else 'FAILURE'}",
            )
            for rollout in selected
        )
        observed: tuple[RolloutMetric, ...] = tuple(
            {
                "seed": capture.seed,
                "steps": len(capture.end_effectors) - 1,
                "success": capture.success,
            }
            for capture in captures
        )
        kept, drifted = filter_replay_outcomes(
            observed,
            expected_success=success,
        )
        if drifted and not args.skip_outcome_drift:
            raise RuntimeError("live rollout outcome disagrees with evaluation metrics")
        kept_seeds = {item["seed"] for item in kept}
        unique_captures = tuple(
            capture for capture in captures if capture.seed in kept_seeds
        )
        if drifted:
            print(
                "excluded outcome-drift seeds "
                + ",".join(str(item["seed"]) for item in drifted),
                flush=True,
            )
        if args.repeat_to_target:
            repeated = repeat_rollouts_to_target(
                kept,
                target_frames=target_frames,
                title_frames=_TITLE_FRAMES,
                hold_frames=hold_frames,
            )
            by_seed = {capture.seed: capture for capture in unique_captures}
            captures = tuple(by_seed[item["seed"]] for item in repeated)
        else:
            captures = unique_captures
        output = assets / reel_filename(spec.model, success=success)
        write_reel(
            captures,
            output,
            hold_seconds=args.hold_seconds,
            hold_frames=hold_frames,
        )
        print(
            f"published {output} with {len(captures)} occurrences "
            f"from {len(unique_captures)} unique cases",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
