"""Detect physical Push-T scene pose from the fixed red goal fiducial."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Image = NDArray[np.uint8]
Contour = NDArray[np.int32]


class _Cv2(Protocol):
    RETR_EXTERNAL: int
    CHAIN_APPROX_SIMPLE: int
    COLOR_BGR2HSV: int

    def findContours(
        self, image: Image, mode: int, method: int
    ) -> tuple[list[Contour], object]: ...
    def contourArea(self, contour: Contour) -> float: ...
    def arcLength(self, curve: Contour, closed: bool) -> float: ...
    def approxPolyDP(self, curve: Contour, epsilon: float, closed: bool) -> Contour: ...
    def findHomography(
        self, source: FloatArray, target: FloatArray, method: int
    ) -> tuple[FloatArray, object]: ...
    def perspectiveTransform(self, source: FloatArray, matrix: FloatArray) -> FloatArray: ...
    def cvtColor(self, source: Image, code: int) -> Image: ...
    def inRange(
        self, source: Image, lower: tuple[int, int, int], upper: tuple[int, int, int]
    ) -> Image: ...


cv2 = cast("_Cv2", import_module("cv2"))
_LOCAL_T = np.array(
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
_TARGET_T = _LOCAL_T + np.array([0.34, 0.0], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TaskSceneDetection:
    block_x_m: float
    block_y_m: float
    block_yaw_rad: float
    pusher_x_m: float
    pusher_y_m: float
    registration_rmse_px: float
    block_fit_rmse_m: float


def _contour(mask: Image, label: str) -> FloatArray:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"{label} T shape is absent")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 500.0:
        raise ValueError(f"{label} T shape is too small")
    perimeter = cv2.arcLength(contour, True)
    for fraction in (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035):
        polygon = cv2.approxPolyDP(contour, fraction * perimeter, True)
        points = np.asarray(polygon[:, 0, :], dtype=np.float64)
        if points.shape == (8, 2):
            return points
    raise ValueError(f"{label} T shape must expose eight corners")


def _best_homography(image_points: FloatArray) -> tuple[FloatArray, float]:
    best_matrix: FloatArray | None = None
    best_rmse = math.inf
    for reversed_order in (False, True):
        ordered = image_points[::-1] if reversed_order else image_points
        for offset in range(8):
            candidate = np.roll(ordered, offset, axis=0)
            matrix, _mask = cv2.findHomography(_TARGET_T, candidate, 0)
            basis = cv2.perspectiveTransform(
                np.array([[[0.34, 0.0], [0.35, 0.0], [0.34, 0.01]]], dtype=np.float64),
                matrix,
            )[0]
            x_axis = basis[1] - basis[0]
            y_axis = basis[2] - basis[0]
            if not (x_axis[1] > abs(x_axis[0]) and y_axis[0] < -abs(y_axis[1])):
                continue
            projected = cv2.perspectiveTransform(_TARGET_T[None], matrix)[0]
            delta = projected - candidate
            rmse = math.sqrt(sum(float(value) * float(value) for value in delta.flat) / delta.size)
            if rmse < best_rmse:
                best_matrix = np.asarray(matrix, dtype=np.float64)
                best_rmse = rmse
    if best_matrix is None or not math.isfinite(best_rmse):
        raise ValueError("red goal homography fit failed")
    return best_matrix, best_rmse


def _fit_block(points: FloatArray) -> tuple[float, float, float, float]:
    best: tuple[float, FloatArray, FloatArray] | None = None
    local_centered = _LOCAL_T - _LOCAL_T.mean(axis=0)
    for reversed_order in (False, True):
        ordered = points[::-1] if reversed_order else points
        for offset in range(8):
            candidate = np.roll(ordered, offset, axis=0)
            centered = candidate - candidate.mean(axis=0)
            left, _singular, right = cast(
                "tuple[FloatArray, FloatArray, FloatArray]",
                np.linalg.svd(local_centered.T @ centered),
            )
            rotation = right.T @ left.T
            if np.linalg.det(rotation) < 0.0:
                right[-1, :] *= -1.0
                rotation = right.T @ left.T
            translation = np.asarray(
                candidate.mean(axis=0) - _LOCAL_T.mean(axis=0) @ rotation.T,
                dtype=np.float64,
            )
            fitted = _LOCAL_T @ rotation.T + translation
            rmse = math.sqrt(float(np.mean(np.square(fitted - candidate))))
            if best is None or rmse < best[0]:
                best = (rmse, translation, rotation)
    if best is None:
        raise ValueError("green block pose fit failed")
    rmse, translation, rotation = best
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return float(translation[0]), float(translation[1]), yaw, rmse


def detect_task_scene(image: Image) -> TaskSceneDetection:
    """Fit task-plane homography and green block pose from one BGR frame."""
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError("scene frame must be uint8[480,640,3]")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 70, 50), (15, 255, 255)) | cv2.inRange(
        hsv, (165, 70, 50), (179, 255, 255)
    )
    green = cv2.inRange(hsv, (35, 45, 25), (95, 255, 255))
    purple = cv2.inRange(hsv, (115, 35, 20), (170, 255, 255))
    red_points = _contour(red, "red goal")
    green_points = _contour(green, "green block")
    task_to_image, registration_rmse = _best_homography(red_points)
    image_to_task = np.asarray(np.linalg.inv(task_to_image), dtype=np.float64)
    block_points = cv2.perspectiveTransform(green_points[None], image_to_task)[0]
    block_x, block_y, block_yaw, block_rmse = _fit_block(block_points)
    purple_contours, _hierarchy = cv2.findContours(
        purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not purple_contours:
        raise ValueError("purple pusher is absent")
    arm = max(purple_contours, key=cv2.contourArea)
    arm_points = np.asarray(arm[:, 0, :], dtype=np.float64)
    tip = arm_points[int(np.argmax(arm_points[:, 1]))]
    pusher = cv2.perspectiveTransform(tip.reshape(1, 1, 2), image_to_task)[0, 0]
    if registration_rmse > 2.5 or block_rmse > 0.012:
        raise ValueError("task fiducial scene fit exceeds residual limits")
    return TaskSceneDetection(
        block_x,
        block_y,
        block_yaw,
        float(pusher[0]),
        float(pusher[1]),
        registration_rmse,
        block_rmse,
    )
