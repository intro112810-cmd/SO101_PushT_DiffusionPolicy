"""Immutable printable checkerboard and guided registration placement contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Final

TARGET_SCHEMA: Final = "so101-camera-registration-charuco-v1"
ARUCO_DICTIONARY: Final = "DICT_5X5_100"
SQUARE_SIZE_M: Final = 0.025
SQUARE_SIZE_MM: Final = 25.0
MARKER_SIZE_M: Final = 0.018
MARKER_SIZE_MM: Final = 18.0
BOARD_COLUMNS: Final = 8
BOARD_ROWS: Final = 6
INNER_COLUMNS: Final = BOARD_COLUMNS - 1
INNER_ROWS: Final = BOARD_ROWS - 1
PRINT_TOLERANCE_MM: Final = 0.125
TARGET_ASSET: Final = Path("docs/assets/camera_registration_charuco_board_only_a4.pdf")


@dataclass(frozen=True, slots=True)
class Placement:
    capture_id: str
    phase: str
    role: str | None
    instruction: str
    table_origin_xy_m: tuple[float, float] | None = None
    table_angle_degrees: int | None = None


INTRINSIC_PLACEMENTS: Final = (
    Placement("intrinsic-01", "intrinsics", None, "centered, board parallel to image plane"),
    Placement("intrinsic-02", "intrinsics", None, "upper-left, tilt top edge away from camera"),
    Placement("intrinsic-03", "intrinsics", None, "upper-right, tilt left edge away from camera"),
    Placement("intrinsic-04", "intrinsics", None, "lower-left, tilt right edge away from camera"),
    Placement("intrinsic-05", "intrinsics", None, "lower-right, tilt bottom edge away from camera"),
    Placement(
        "intrinsic-06", "intrinsics", None, "centered close view, compound horizontal/vertical tilt"
    ),
)
TABLE_PLACEMENTS: Final = (
    Placement(
        "table-fit-a",
        "table_fit",
        "calibration_fit",
        "flat; outer top-left at (-0.200,-0.150) m; top edge along physical +X",
        (-0.2, -0.15),
        0,
    ),
    Placement(
        "table-fit-b",
        "table_fit",
        "calibration_fit",
        "flat; outer top-left at (+0.000,-0.150) m; top edge along physical +X",
        (0.0, -0.15),
        0,
    ),
    Placement(
        "table-fit-c",
        "table_fit",
        "calibration_fit",
        "flat; outer top-left at (-0.200,+0.000) m; top edge along physical +X",
        (-0.2, 0.0),
        0,
    ),
    Placement(
        "checkpoint-held-a",
        "checkpoint",
        "checkpoint_held_out",
        "flat at (+0.000,+0.000) m, top edge physical +X; arm parked; T block at (-0.050,0.000) m",
        (0.0, 0.0),
        0,
    ),
    Placement(
        "checkpoint-held-b",
        "checkpoint",
        "checkpoint_held_out",
        "flat at (-0.100,-0.075) m, top edge physical +X; arm parked; T block at (+0.050,+0.050) m",
        (-0.1, -0.075),
        0,
    ),
)
PLACEMENTS: Final = INTRINSIC_PLACEMENTS + TABLE_PLACEMENTS


def target_document() -> dict[str, object]:
    """Return the machine-consumed physical target contract."""
    return {
        "schema": TARGET_SCHEMA,
        "target_type": "charuco",
        "aruco_dictionary": ARUCO_DICTIONARY,
        "board_squares": [BOARD_COLUMNS, BOARD_ROWS],
        "inner_corners": [INNER_COLUMNS, INNER_ROWS],
        "square_size_mm": SQUARE_SIZE_MM,
        "marker_size_mm": MARKER_SIZE_MM,
        "board_size_mm": [BOARD_COLUMNS * SQUARE_SIZE_MM, BOARD_ROWS * SQUARE_SIZE_MM],
        "accepted_square_error_mm": PRINT_TOLERANCE_MM,
        "corner_order": "row_major_from_first_inner_corner_nearest_outer_top_left",
    }


def target_digest() -> str:
    encoded = json.dumps(target_document(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_print_scale(measured_square_mm: float) -> None:
    """Reject nonfinite or wrongly scaled printouts before any camera opens."""
    if not math.isfinite(measured_square_mm) or (
        abs(measured_square_mm - SQUARE_SIZE_MM) > PRINT_TOLERANCE_MM
    ):
        raise ValueError("print scale invalid: 25 mm square must measure 24.875 through 25.125 mm")


def local_inner_corners() -> list[tuple[float, float, float]]:
    """Known target coordinates in metres, excluding the outer square border."""
    return [
        ((column + 1) * SQUARE_SIZE_M, (row + 1) * SQUARE_SIZE_M, 0.0)
        for row in range(INNER_ROWS)
        for column in range(INNER_COLUMNS)
    ]


def table_corners(placement: Placement) -> list[tuple[float, float, float]]:
    """Bind target-local corners to one explicit physical-table placement."""
    if placement.table_origin_xy_m is None or placement.table_angle_degrees is None:
        raise ValueError("placement does not bind the target to the table frame")
    angle = math.radians(placement.table_angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    origin_x, origin_y = placement.table_origin_xy_m
    return [
        (
            origin_x + cosine * x - sine * y,
            origin_y + sine * x + cosine * y,
            z,
        )
        for x, y, z in local_inner_corners()
    ]


def placement_digest() -> str:
    payload = [
        {
            "capture_id": item.capture_id,
            "phase": item.phase,
            "role": item.role,
            "instruction": item.instruction,
            "table_origin_xy_m": item.table_origin_xy_m,
            "table_angle_degrees": item.table_angle_degrees,
        }
        for item in PLACEMENTS
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
