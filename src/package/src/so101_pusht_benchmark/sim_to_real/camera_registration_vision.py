"""OpenCV corner detection and raw registration geometry fitting."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from typing import cast, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from .task_frame import Se2
from .camera_registration_target import (
    ARUCO_DICTIONARY,
    BOARD_COLUMNS,
    BOARD_ROWS,
    INNER_COLUMNS,
    INNER_ROWS,
    MARKER_SIZE_M,
    Placement,
    SQUARE_SIZE_M,
    local_inner_corners,
    table_corners,
)

Float32Array = NDArray[np.float32]
FloatArray = NDArray[np.float64]
Image = NDArray[np.uint8]
MIN_BLUR_VARIANCE = 80.0
MIN_POSE_DISTANCE = 0.035
DETECTION_SCALE: Final = 3


class _CharucoDetector(Protocol):
    def detectBoard(
        self, image: NDArray[np.uint8]
    ) -> tuple[
        NDArray[np.float32] | None,
        NDArray[np.int32] | None,
        list[NDArray[np.float32]],
        NDArray[np.int32] | None,
    ]: ...


class _Aruco(Protocol):
    DICT_5X5_100: int

    def getPredefinedDictionary(self, dictionary: int) -> object: ...

    def CharucoBoard(
        self, size: tuple[int, int], square_length: float, marker_length: float, dictionary: object
    ) -> object: ...

    def CharucoDetector(self, board: object) -> _CharucoDetector: ...


class _Cv2(Protocol):
    aruco: _Aruco
    COLOR_BGR2GRAY: int
    INTER_CUBIC: int
    TERM_CRITERIA_EPS: int
    TERM_CRITERIA_MAX_ITER: int
    SOLVEPNP_ITERATIVE: int

    def cvtColor(self, source: Image, code: int) -> NDArray[np.uint8]: ...

    def resize(
        self, source: NDArray[np.uint8], size: tuple[int, int], interpolation: int
    ) -> NDArray[np.uint8]: ...

    def Laplacian(self, source: NDArray[np.uint8], depth: int) -> NDArray[np.float64]: ...

    def calibrateCamera(
        self,
        object_points: list[Float32Array],
        image_points: list[Float32Array],
        size: tuple[int, int],
        camera_matrix: None,
        distortion: None,
    ) -> tuple[float, FloatArray, FloatArray, list[FloatArray], list[FloatArray]]: ...

    def solvePnP(
        self,
        object_points: FloatArray,
        image_points: FloatArray,
        camera_matrix: FloatArray,
        distortion: FloatArray,
        flags: int = 0,
    ) -> tuple[bool, FloatArray, FloatArray]: ...

    def Rodrigues(self, vector: FloatArray) -> tuple[FloatArray, FloatArray]: ...

    def imencode(self, extension: str, image: Image) -> tuple[bool, NDArray[np.uint8]]: ...


@dataclass(frozen=True, slots=True)
class DetectedView:
    placement: Placement
    corners_px: FloatArray


@dataclass(frozen=True, slots=True)
class FittedGeometry:
    intrinsics: tuple[float, ...]
    distortion: tuple[float, ...]
    camera_to_table: tuple[float, ...]
    physical_to_sim: Se2


def _cv2() -> _Cv2:
    return cast("_Cv2", import_module("cv2"))


def detect_checkerboard(frame: Image) -> tuple[FloatArray, float]:
    """Detect all ID-bound 7x5 ChArUco corners and reject blurred frames."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("camera frame must be uint8 BGR")
    cv2 = _cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, 6).var())
    if not math.isfinite(blur_variance) or blur_variance < MIN_BLUR_VARIANCE:
        raise ValueError("frame is too blurred for registration")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, ARUCO_DICTIONARY):
        raise RuntimeError("installed OpenCV lacks the required aruco dictionary")
    dictionary_id = cv2.aruco.DICT_5X5_100
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLUMNS, BOARD_ROWS), SQUARE_SIZE_M, MARKER_SIZE_M, dictionary
    )
    detection_gray = cv2.resize(
        gray,
        (gray.shape[1] * DETECTION_SCALE, gray.shape[0] * DETECTION_SCALE),
        interpolation=cv2.INTER_CUBIC,
    )
    raw, raw_ids, _marker_corners, _marker_ids = cv2.aruco.CharucoDetector(board).detectBoard(
        detection_gray
    )
    count = INNER_COLUMNS * INNER_ROWS
    if raw is None or raw_ids is None:
        raise ValueError("ChArUco corner detection failed")
    detected = np.asarray(raw, dtype=np.float64).reshape(-1, 2) / DETECTION_SCALE
    ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
    if (
        detected.shape != (count, 2)
        or {int(value) for value in ids} != set(range(count))
        or not np.isfinite(detected).all()
    ):
        raise ValueError("ChArUco corner detection is incomplete")
    corners = np.empty((count, 2), dtype=np.float64)
    corners[ids] = detected
    return corners, blur_variance


def pose_distance(left: FloatArray, right: FloatArray, resolution: tuple[int, int]) -> float:
    """Return normalized RMS image-corner displacement for repeat rejection."""
    diagonal = math.hypot(*resolution)
    return float(np.sqrt(np.mean(np.square(left - right))) / diagonal)


def reject_repeated_pose(
    candidate: FloatArray,
    accepted: list[DetectedView],
    resolution: tuple[int, int],
) -> None:
    if any(
        pose_distance(candidate, view.corners_px, resolution) < MIN_POSE_DISTANCE
        for view in accepted
    ):
        raise ValueError("repeated target pose; move and tilt to the instructed placement")


def encode_png(frame: Image) -> bytes:
    success, encoded = _cv2().imencode(".png", frame)
    if not success:
        raise RuntimeError("raw PNG encoding failed")
    return encoded.tobytes()


def _object_points() -> FloatArray:
    return np.asarray(local_inner_corners(), dtype=np.float64)


def _fit_intrinsics(
    views: list[DetectedView], resolution: tuple[int, int]
) -> tuple[FloatArray, FloatArray]:
    intrinsic = [view for view in views if view.placement.phase == "intrinsics"]
    if len(intrinsic) < 6:
        raise ValueError("at least six nondegenerate intrinsic views are required")
    cv2 = _cv2()
    object_points = [
        np.ascontiguousarray(_object_points(), dtype=np.float32) for _view in intrinsic
    ]
    image_points = [
        np.ascontiguousarray(view.corners_px.reshape(-1, 1, 2), dtype=np.float32)
        for view in intrinsic
    ]
    rms, matrix, distortion, _rotations, _translations = cv2.calibrateCamera(
        object_points, image_points, resolution, None, None
    )
    matrix = np.asarray(matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if (
        not math.isfinite(float(rms))
        or matrix.shape != (3, 3)
        or distortion.size < 5
        or not np.isfinite(matrix).all()
        or not np.isfinite(distortion[:5]).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
    ):
        raise ValueError("intrinsic calibration produced invalid geometry")
    return matrix, distortion[:5]


def fit_intrinsic_views(
    views: list[DetectedView], resolution: tuple[int, int]
) -> tuple[FloatArray, FloatArray]:
    """Expose the current governed intrinsic fit without table-registration finalization."""
    return _fit_intrinsics(views, resolution)


def _fit_camera_to_table(
    views: list[DetectedView], intrinsics: FloatArray, distortion: FloatArray
) -> tuple[float, ...]:
    fit = [view for view in views if view.placement.role == "calibration_fit"]
    if len(fit) < 3:
        raise ValueError("at least three table-plane fit views are required")
    world = np.concatenate(
        [np.asarray(table_corners(view.placement), dtype=np.float64) for view in fit]
    )
    image = np.concatenate([view.corners_px for view in fit])
    success, rotation_vector, translation = _cv2().solvePnP(
        world,
        image,
        intrinsics,
        distortion,
        flags=_cv2().SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError("camera-to-table fit failed")
    table_to_camera, _jacobian = _cv2().Rodrigues(rotation_vector)
    table_to_camera = np.asarray(table_to_camera, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    camera_to_table = table_to_camera.T
    camera_origin_table = -(camera_to_table @ translation)
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3, :3] = camera_to_table
    homogeneous[:3, 3] = camera_origin_table
    if not np.isfinite(homogeneous).all():
        raise ValueError("camera-to-table fit produced nonfinite geometry")
    return tuple(float(value) for value in homogeneous.reshape(-1))


def _fit_physical_to_sim(views: list[DetectedView]) -> Se2:
    table_views = [view for view in views if view.placement.role is not None]
    physical = np.concatenate(
        [np.asarray(table_corners(view.placement), dtype=np.float64)[:, :2] for view in table_views]
    )
    # The printed placement protocol defines matching physical/simulator origin and axes.
    simulation = physical.copy()
    physical_center = physical.mean(axis=0)
    simulation_center = simulation.mean(axis=0)
    covariance = (physical - physical_center).T @ (simulation - simulation_center)
    left, _singular, right = cast(
        "tuple[FloatArray, FloatArray, FloatArray]", np.linalg.svd(covariance)
    )
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1, :] *= -1
        rotation = right.T @ left.T
    translation = simulation_center - rotation @ physical_center
    if not np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-9) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9
    ):
        raise ValueError("physical-to-simulation fit is not rigid SE(2)")
    return (
        float(rotation[0, 0]),
        float(rotation[0, 1]),
        float(translation[0]),
        float(rotation[1, 0]),
        float(rotation[1, 1]),
        float(translation[1]),
    )


def fit_geometry(views: list[DetectedView], resolution: tuple[int, int]) -> FittedGeometry:
    """Fit only from detected pixels and the immutable physical target contract."""
    if len({view.placement.capture_id for view in views}) != len(views):
        raise ValueError("capture identities must be unique")
    intrinsic, distortion = _fit_intrinsics(views, resolution)
    return FittedGeometry(
        tuple(float(value) for value in intrinsic.reshape(-1)),
        tuple(float(value) for value in distortion),
        _fit_camera_to_table(views, intrinsic, distortion),
        _fit_physical_to_sim(views),
    )
