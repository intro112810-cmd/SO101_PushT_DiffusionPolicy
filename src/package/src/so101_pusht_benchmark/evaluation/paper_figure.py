"""Pure geometry and clustering helpers for the paper-style Push-T figure."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def time_gradient(point_count: int) -> FloatArray:
    """Return one color per line segment, purple at start and yellow at end."""
    if point_count < 2:
        raise ValueError("a trajectory requires at least two points")
    start = np.asarray([0.050, 0.030, 0.528, 1.0], dtype=np.float64)
    end = np.asarray([0.940, 0.975, 0.131, 1.0], dtype=np.float64)
    progress = np.linspace(
        0.0,
        1.0,
        point_count - 1,
        dtype=np.float64,
    )[:, None]
    return np.asarray(start + progress * (end - start), dtype=np.float64)


def trajectory_segments(points: FloatArray) -> FloatArray:
    """Pair consecutive XY points for a matplotlib LineCollection."""
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("trajectory must be float[N,2] with N >= 2")
    return np.stack((points[:-1], points[1:]), axis=1)


def _rectangle(
    center: FloatArray,
    *,
    width: float,
    height: float,
    offset_y: float,
    yaw: float,
) -> FloatArray:
    corners = np.asarray(
        [
            [-width / 2, -height / 2 + offset_y],
            [width / 2, -height / 2 + offset_y],
            [width / 2, height / 2 + offset_y],
            [-width / 2, height / 2 + offset_y],
        ],
        dtype=np.float64,
    )
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return corners @ rotation.T + center


def t_shape_polygons(center: FloatArray, *, yaw: float) -> tuple[FloatArray, FloatArray]:
    """Return top and stem polygons matching the frozen MuJoCo T geometry."""
    if center.shape != (2,):
        raise ValueError("T center must be float[2]")
    return (
        _rectangle(center, width=0.10, height=0.03, offset_y=0.035, yaw=yaw),
        _rectangle(center, width=0.03, height=0.07, offset_y=-0.015, yaw=yaw),
    )


def route_mode(
    trajectory: FloatArray,
    block_center: FloatArray,
    *,
    contact_radius: float = 0.075,
    minimum_angle: float = 0.35,
    minimum_commitment: float = 0.80,
) -> int:
    """Classify a committed clockwise/counter-clockwise route around the block."""
    if trajectory.ndim != 2 or trajectory.shape[1] != 2 or len(trajectory) < 8:
        raise ValueError("trajectory must be float[N,2] with N >= 8")
    if block_center.shape != (2,):
        raise ValueError("block center must be float[2]")
    vectors = trajectory - block_center
    distances = np.linalg.norm(vectors, axis=1)
    closest = int(np.argmin(distances[5:]) + 5)
    if distances[closest] > contact_radius:
        return 0
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    deltas = angles - angles[0]
    angular_change = float(deltas[closest])
    if abs(angular_change) < minimum_angle:
        return 0
    direction = 1 if angular_change > 0 else -1
    meaningful = deltas[1 : closest + 1]
    meaningful = meaningful[np.abs(meaningful) >= 0.10]
    if len(meaningful) == 0:
        return 0
    commitment = np.count_nonzero(np.sign(meaningful) == direction) / len(
        meaningful
    )
    return direction if commitment >= minimum_commitment else 0


def classify_route_behavior(route_modes: list[int], *, sample_count: int) -> str:
    """Name only behavior supported by committed clockwise/counter-clockwise counts."""
    if len(route_modes) != sample_count or sample_count < 4:
        raise ValueError("route modes must match a sample count of at least four")
    clockwise = route_modes.count(-1)
    counter_clockwise = route_modes.count(1)
    if min(clockwise, counter_clockwise) >= 2:
        return "multimodal"
    if min(clockwise, counter_clockwise) == 0 and max(
        clockwise,
        counter_clockwise,
    ) >= sample_count // 2:
        return "single-mode"
    return "uncommitted"


def two_mode_score(trajectories: FloatArray) -> tuple[float, NDArray[np.int64]]:
    """Score a balanced deterministic two-cluster partition of XY trajectories."""
    if trajectories.ndim != 3 or trajectories.shape[2] != 2 or len(trajectories) < 4:
        raise ValueError("trajectories must be float[N,T,2] with N >= 4")
    normalized = trajectories - trajectories[:, :1]
    features = normalized.reshape(len(normalized), -1)
    distances = np.linalg.norm(features[:, None] - features[None, :], axis=2)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    centroids = np.stack((features[first], features[second]))
    labels = np.zeros(len(features), dtype=np.int64)
    for _ in range(20):
        updated = np.argmin(
            np.linalg.norm(features[:, None] - centroids[None, :], axis=2),
            axis=1,
        )
        if np.array_equal(updated, labels) and np.count_nonzero(labels) > 0:
            break
        labels = updated
        if any(np.count_nonzero(labels == index) == 0 for index in (0, 1)):
            return 0.0, labels
        centroids = np.stack(
            [features[labels == index].mean(axis=0) for index in (0, 1)]
        )
    cluster_sizes = np.asarray(
        [np.count_nonzero(labels == index) for index in (0, 1)],
        dtype=np.float64,
    )
    balance = cluster_sizes.min() / cluster_sizes.max()
    separation = float(np.linalg.norm(centroids[0] - centroids[1]))
    within = float(
        np.mean(
            [
                np.linalg.norm(features[index] - centroids[labels[index]])
                for index in range(len(features))
            ]
        )
    )
    return separation * balance / (1.0 + within), labels
