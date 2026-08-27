"""Offline video decoding, intrinsic fitting, and deterministic image rendering."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from . import camera_registration_vision as registration_vision
from .camera_registration_target import Placement, local_inner_corners
from .camera_registration_vision import DetectedView, encode_png
from .intrinsic_extraction import ExtractionError
from .intrinsic_extraction_types import (
    Corners,
    DecodedFrame,
    Decoder,
    FitEvaluation,
    FitQuality,
    HeldoutMetrics,
    Image,
)


class _CalibrationCv2(Protocol):
    SOLVEPNP_ITERATIVE: int

    def solvePnP(
        self,
        object_points: NDArray[np.float64],
        image_points: NDArray[np.float64],
        camera_matrix: NDArray[np.float64],
        distortion: NDArray[np.float64],
        flags: int = 0,
    ) -> tuple[bool, NDArray[np.float64], NDArray[np.float64]]: ...

    def projectPoints(
        self,
        object_points: NDArray[np.float64],
        rotation: NDArray[np.float64],
        translation: NDArray[np.float64],
        camera_matrix: NDArray[np.float64],
        distortion: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@dataclass(frozen=True, slots=True)
class _ProjectionModel:
    cv2: _CalibrationCv2
    object_points: NDArray[np.float64]
    matrix: NDArray[np.float64]
    distortion: NDArray[np.float64]


def _reprojection_errors(
    corners: tuple[Corners, ...], model: _ProjectionModel
) -> NDArray[np.float64]:
    errors: list[float] = []
    for detected in corners:
        success, rotation, translation = model.cv2.solvePnP(
            model.object_points,
            detected.reshape(-1, 1, 2),
            model.matrix,
            model.distortion,
            flags=model.cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise ExtractionError("intrinsic reprojection solve failed")
        projected, _jacobian = model.cv2.projectPoints(
            model.object_points,
            rotation,
            translation,
            model.matrix,
            model.distortion,
        )
        errors.extend(
            float(value) for value in np.linalg.norm(projected.reshape(-1, 2) - detected, axis=1)
        )
    return np.asarray(errors, dtype=np.float64)


def decode_video(path: Path) -> Iterator[DecodedFrame]:
    """Yield every native video frame sequentially with exact stream PTS metadata."""
    with closing(av.open(path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or frame.time_base.denominator <= 0:
                raise ExtractionError("decoded frame lacks a finite native presentation timestamp")
            timestamp = float(frame.pts * frame.time_base)
            if not math.isfinite(timestamp):
                raise ExtractionError("decoded frame has a nonfinite native presentation timestamp")
            yield DecodedFrame(
                frame_index,
                frame.pts,
                frame.time_base.numerator,
                frame.time_base.denominator,
                timestamp,
                frame.to_ndarray(format="bgr24"),
            )


def calibrate_and_evaluate(
    fit_corners: tuple[Corners, ...],
    heldout_corners: tuple[Corners, ...],
    resolution: tuple[int, int],
) -> FitEvaluation:
    """Run the current fit and independently score unseen held-out corners."""
    views = [
        DetectedView(Placement(f"offline-fit-{index:02d}", "intrinsics", None, "offline"), detected)
        for index, detected in enumerate(fit_corners)
    ]
    matrix, distortion = registration_vision.fit_intrinsic_views(views, resolution)
    calibration_cv2: _CalibrationCv2 = cv2
    model = _ProjectionModel(
        calibration_cv2,
        np.asarray(local_inner_corners(), dtype=np.float64),
        matrix,
        distortion,
    )
    fit_errors = _reprojection_errors(fit_corners, model)
    heldout_errors = _reprojection_errors(heldout_corners, model)
    heldout = HeldoutMetrics(
        float(np.sqrt(np.mean(np.square(heldout_errors)))),
        float(heldout_errors.mean()),
        float(np.median(heldout_errors)),
        float(np.percentile(heldout_errors, 95)),
        float(heldout_errors.max()),
        int(heldout_errors.size),
    )
    if not all(
        math.isfinite(value)
        for value in (
            float(np.sqrt(np.mean(np.square(fit_errors)))),
            heldout.rms_error_px,
            heldout.mean_error_px,
            heldout.median_error_px,
            heldout.p95_error_px,
            heldout.max_error_px,
        )
    ):
        raise ExtractionError("held-out intrinsic reprojection metrics are nonfinite")
    return FitEvaluation(
        FitQuality(
            float(np.sqrt(np.mean(np.square(fit_errors)))),
            tuple(float(value) for value in matrix.reshape(-1)),
            tuple(float(value) for value in distortion.reshape(-1)),
        ),
        heldout,
    )


def selected_images(
    source: Path,
    selected_indices: tuple[int, ...],
    decode: Decoder = decode_video,
) -> tuple[Image, ...]:
    """Re-decode sequentially and return only the six selected original frames."""
    requested = set(selected_indices)
    images: dict[int, Image] = {}
    for frame in decode(source):
        if frame.frame_index in requested:
            images[frame.frame_index] = frame.image
    missing = requested.difference(images)
    if missing:
        raise ExtractionError(f"selected frames missing during evidence decode: {sorted(missing)}")
    return tuple(images[index] for index in selected_indices)


def png_bytes(image: Image) -> bytes:
    return encode_png(image)


def contact_sheets(images: tuple[Image, ...]) -> tuple[bytes, ...]:
    """Render deterministic three-column by two-row sheets for all selected images."""
    sheets: list[bytes] = []
    for offset in range(0, len(images), 6):
        chunk = images[offset : offset + 6]
        tiles = [cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA) for image in chunk]
        tiles.extend(np.zeros((240, 320, 3), dtype=np.uint8) for _index in range(6 - len(tiles)))
        sheet = np.concatenate(
            (np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)),
            axis=0,
        )
        sheets.append(encode_png(sheet))
    return tuple(sheets)
