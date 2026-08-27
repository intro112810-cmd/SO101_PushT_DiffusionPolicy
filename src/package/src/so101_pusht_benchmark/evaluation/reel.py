"""Pure frame-budget helpers for professor-facing rollout reels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict


class RolloutMetric(TypedDict):
    seed: int
    steps: int
    success: bool


def hold_frame_count(*, seconds: float, fps: int) -> int:
    """Convert a visible playback hold duration to encoded frames."""
    if seconds < 0:
        raise ValueError("hold seconds must be non-negative")
    return round(seconds * fps)


def select_reel_rollouts(
    rollouts: Sequence[RolloutMetric],
    *,
    success: bool,
    target_frames: int,
    title_frames: int,
    hold_frames: int,
) -> tuple[RolloutMetric, ...]:
    """Select ordered cases without materially exceeding the target duration."""
    selected: list[RolloutMetric] = []
    used_frames = 0
    for rollout in rollouts:
        if rollout["success"] is not success:
            continue
        case_frames = title_frames + rollout["steps"] + 1 + hold_frames
        if selected and used_frames + case_frames > target_frames:
            break
        selected.append(rollout)
        used_frames += case_frames
    if not selected:
        raise ValueError(f"no {'success' if success else 'failure'} rollout selected")
    return tuple(selected)


def filter_replay_outcomes(
    observed: Sequence[RolloutMetric],
    *,
    expected_success: bool,
) -> tuple[tuple[RolloutMetric, ...], tuple[RolloutMetric, ...]]:
    """Separate reproducible cases from evaluation/replay outcome drift."""
    kept = tuple(
        rollout
        for rollout in observed
        if rollout["success"] is expected_success
    )
    drifted = tuple(
        rollout
        for rollout in observed
        if rollout["success"] is not expected_success
    )
    if not kept:
        raise ValueError("no replay outcome matches the requested group")
    return kept, drifted


def repeat_rollouts_to_target(
    rollouts: Sequence[RolloutMetric],
    *,
    target_frames: int,
    title_frames: int,
    hold_frames: int,
) -> tuple[RolloutMetric, ...]:
    """Cycle validated sparse cases until the encoded reel reaches its target."""
    if not rollouts:
        raise ValueError("cannot repeat an empty rollout sequence")
    repeated = list(rollouts)
    used_frames = sum(
        title_frames + rollout["steps"] + 1 + hold_frames
        for rollout in repeated
    )
    index = 0
    while used_frames < target_frames:
        rollout = rollouts[index % len(rollouts)]
        repeated.append(rollout)
        used_frames += title_frames + rollout["steps"] + 1 + hold_frames
        index += 1
    return tuple(repeated)
