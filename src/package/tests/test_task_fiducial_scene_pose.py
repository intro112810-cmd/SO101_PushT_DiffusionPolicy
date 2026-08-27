"""Task-plane scene pose from the fixed red Push-T goal fiducial."""

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from so101_pusht_benchmark.sim_to_real.task_fiducial_scene_pose import detect_task_scene


def test_detects_green_block_pose_from_red_goal_homography() -> None:
    image = np.full((480, 640, 3), 245, dtype=np.uint8)
    target = np.array(
        [
            [-0.055, -0.014],
            [0.055, -0.014],
            [0.055, 0.014],
            [0.014, 0.014],
            [0.014, 0.072],
            [-0.014, 0.072],
            [-0.014, 0.014],
            [-0.055, 0.014],
        ],
        dtype=np.float64,
    )
    task_to_px = np.array([[0.0, -720.0, 320.0], [690.0, 0.0, -44.6], [0.0, 0.0, 1.0]])

    def project(points: NDArray[np.float64]) -> NDArray[np.int32]:
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ task_to_px.T
        return np.rint(homogeneous[:, :2] / homogeneous[:, 2:]).astype(np.int32)

    cv2.fillPoly(image, [project(target + np.array([0.34, 0.0]))], (40, 40, 210))
    yaw = 0.45
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    block = target @ rotation.T + [0.25, 0.08]
    cv2.fillPoly(image, [project(block)], (40, 170, 50))
    cv2.circle(image, tuple(project(np.array([[0.30, 0.10]]))[0]), 9, (170, 40, 150), -1)

    detected = detect_task_scene(image)

    assert abs(detected.block_x_m - 0.25) < 0.012
    assert abs(detected.block_y_m - 0.08) < 0.01
    assert abs(detected.block_yaw_rad - yaw) < 0.08
    assert detected.registration_rmse_px < 2.5
    assert detected.block_fit_rmse_m < 0.012
