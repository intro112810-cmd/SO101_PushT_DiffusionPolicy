from __future__ import annotations

from so101_pusht_benchmark.evaluation.reel import (
    RolloutMetric,
    filter_replay_outcomes,
    hold_frame_count,
    repeat_rollouts_to_target,
    select_reel_rollouts,
)


def test_hold_frame_count_uses_playback_seconds() -> None:
    assert hold_frame_count(seconds=2.0, fps=40) == 80


def test_select_reel_rollouts_respects_outcome_and_frame_budget() -> None:
    rollouts: list[RolloutMetric] = [
        {"seed": 1, "steps": 100, "success": True},
        {"seed": 2, "steps": 200, "success": True},
        {"seed": 3, "steps": 300, "success": True},
        {"seed": 4, "steps": 100, "success": False},
    ]

    selected = select_reel_rollouts(
        rollouts,
        success=True,
        target_frames=650,
        title_frames=20,
        hold_frames=80,
    )

    assert [item["seed"] for item in selected] == [1, 2]


def test_filter_replay_outcomes_keeps_only_reproducible_group_members() -> None:
    observed: list[RolloutMetric] = [
        {"seed": 100002, "steps": 300, "success": False},
        {"seed": 100070, "steps": 283, "success": True},
    ]

    kept, drifted = filter_replay_outcomes(observed, expected_success=False)

    assert [item["seed"] for item in kept] == [100002]
    assert [item["seed"] for item in drifted] == [100070]


def test_repeat_rollouts_to_target_cycles_sparse_outcomes_transparently() -> None:
    sparse: list[RolloutMetric] = [
        {"seed": 1, "steps": 100, "success": True},
        {"seed": 2, "steps": 100, "success": True},
    ]

    repeated = repeat_rollouts_to_target(
        sparse,
        target_frames=600,
        title_frames=20,
        hold_frames=80,
    )

    assert [item["seed"] for item in repeated] == [1, 2, 1]

