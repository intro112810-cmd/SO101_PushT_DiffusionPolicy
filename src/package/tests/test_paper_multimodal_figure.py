from __future__ import annotations

import numpy as np

from so101_pusht_benchmark.evaluation.paper_figure import (
    classify_route_behavior,
    route_mode,
    t_shape_polygons,
    time_gradient,
    trajectory_segments,
    two_mode_score,
)


def test_time_gradient_runs_from_purple_to_yellow() -> None:
    colors = time_gradient(5)

    assert colors.shape == (4, 4)
    assert colors[0, 2] > colors[0, 0]
    assert colors[-1, 0] > colors[-1, 2]


def test_trajectory_segments_pair_consecutive_points() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 5.0]])

    segments = trajectory_segments(points)

    np.testing.assert_allclose(
        segments,
        np.asarray([[[0.0, 0.0], [1.0, 2.0]], [[1.0, 2.0], [3.0, 5.0]]]),
    )


def test_t_shape_polygons_match_simulator_dimensions() -> None:
    top, stem = t_shape_polygons(np.asarray([0.0, 0.0]), yaw=0.0)

    assert np.ptp(top[:, 0]) == 0.10
    assert np.ptp(top[:, 1]) == 0.03
    assert np.ptp(stem[:, 0]) == 0.03
    assert np.ptp(stem[:, 1]) == 0.07


def test_two_mode_score_prefers_balanced_separated_paths() -> None:
    upper = np.stack(
        [np.column_stack((np.linspace(0, 1, 10), np.linspace(0, value, 10))) for value in (0.8, 1.0, 1.2)]
    )
    lower = np.stack(
        [np.column_stack((np.linspace(0, 1, 10), np.linspace(0, value, 10))) for value in (-0.8, -1.0, -1.2)]
    )
    separated = np.concatenate((upper, lower))
    collapsed = np.stack(
        [
            np.column_stack((np.linspace(0, 1, 10), np.linspace(0, value, 10)))
            for value in np.linspace(0.0, 0.1, 6)
        ]
    )

    separated_score, labels = two_mode_score(separated)
    collapsed_score, _ = two_mode_score(collapsed)

    assert separated_score > collapsed_score * 5
    assert np.count_nonzero(labels == 0) == 3
    assert np.count_nonzero(labels == 1) == 3


def test_route_mode_requires_opposite_angular_routes_to_block() -> None:
    center = np.asarray([0.0, 0.0])
    radii = np.linspace(0.20, 0.07, 40)
    upper_angles = np.linspace(0.0, 1.2, 40)
    lower_angles = np.linspace(0.0, -1.2, 40)
    upper = np.column_stack(
        (radii * np.cos(upper_angles), radii * np.sin(upper_angles))
    )
    lower = np.column_stack(
        (radii * np.cos(lower_angles), radii * np.sin(lower_angles))
    )
    direct = np.column_stack((radii, np.zeros_like(radii)))

    assert route_mode(upper, center) == 1
    assert route_mode(lower, center) == -1
    assert route_mode(direct, center) == 0


def test_route_behavior_claim_matches_committed_mode_counts() -> None:
    assert classify_route_behavior([-1, -1, 1, 1], sample_count=4) == "multimodal"
    assert classify_route_behavior([1, 1, 1, 0], sample_count=4) == "single-mode"
    assert classify_route_behavior([1, -1, 0, 0], sample_count=4) == "uncommitted"
