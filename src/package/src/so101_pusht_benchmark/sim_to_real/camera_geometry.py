"""Raw pinhole geometry recomputation for camera-registration evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import cast

from .rollout_codes import RolloutCode, RolloutViolation
from .task_frame import parse_rigid_se2


@dataclass(frozen=True, slots=True)
class CameraModel:
    intrinsics: tuple[float, ...]
    distortion: tuple[float, ...]
    camera_to_table: tuple[float, ...]
    physical_to_sim: tuple[float, ...]


def invalid(detail: str) -> RolloutViolation:
    return RolloutViolation(RolloutCode.CAMERA_UNREGISTERED, detail)


def mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise invalid(f"{label} must be a mapping")
    return cast("dict[str, object]", value)


def sequence(value: object, label: str, length: int | None = None) -> list[object]:
    if not isinstance(value, list):
        suffix = "" if length is None else f"[{length}]"
        raise invalid(f"{label} must be a sequence{suffix}")
    result = cast("list[object]", value)
    if length is not None and len(result) != length:
        raise invalid(f"{label} must be a sequence[{length}]")
    return result


def number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise invalid(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise invalid(f"{label} must be finite")
    return result


def vector(value: object, label: str, length: int) -> tuple[float, ...]:
    return tuple(number(item, label) for item in sequence(value, label, length))


def _exact_fields(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(raw) != expected:
        raise invalid(f"{label} fields are incomplete or unknown")


def _parse_intrinsics(value: object, resolution: tuple[int, int]) -> tuple[float, ...]:
    raw = mapping(value, "intrinsics")
    _exact_fields(raw, {"model", "matrix", "units"}, "intrinsics")
    if raw["model"] != "pinhole_brown_conrady" or raw["units"] != "pixels":
        raise invalid("intrinsics model or units are invalid")
    matrix = vector(raw["matrix"], "intrinsics.matrix", 9)
    fx, skew, cx, zero, fy, cy, bottom0, bottom1, bottom2 = matrix
    if fx <= 0 or fy <= 0 or (skew, zero, bottom0, bottom1, bottom2) != (0, 0, 0, 0, 1):
        raise invalid("intrinsics matrix is not canonical pinhole geometry")
    if not (0 <= cx < resolution[0] and 0 <= cy < resolution[1]):
        raise invalid("intrinsics principal point is outside the image")
    return matrix


def _parse_distortion(value: object) -> tuple[float, ...]:
    raw = mapping(value, "distortion")
    _exact_fields(raw, {"model", "coefficients", "order"}, "distortion")
    if raw["model"] != "brown_conrady" or raw["order"] != ["k1", "k2", "p1", "p2", "k3"]:
        raise invalid("distortion model or coefficient order is invalid")
    coefficients = vector(raw["coefficients"], "distortion.coefficients", 5)
    if any(abs(item) > 1.0 for item in coefficients):
        raise invalid("distortion coefficients are outside the supported model domain")
    return coefficients


def _parse_extrinsics(value: object) -> tuple[float, ...]:
    raw = mapping(value, "camera_to_table")
    _exact_fields(
        raw,
        {"direction", "matrix", "translation_units", "camera_axes", "table_axes"},
        "camera_to_table",
    )
    if (
        raw["direction"] != "camera_to_table"
        or raw["translation_units"] != "meters"
        or raw["camera_axes"] != "x_right_y_down_z_forward"
        or raw["table_axes"] != "x_right_y_forward_z_up"
    ):
        raise invalid("camera extrinsic units, axes, or frame direction are invalid")
    matrix = vector(raw["matrix"], "camera_to_table.matrix", 16)
    if matrix[12:] != (0, 0, 0, 1):
        raise invalid("camera_to_table homogeneous row is invalid")
    rotation = (
        matrix[0],
        matrix[1],
        matrix[2],
        matrix[4],
        matrix[5],
        matrix[6],
        matrix[8],
        matrix[9],
        matrix[10],
    )
    rows = (rotation[0:3], rotation[3:6], rotation[6:9])
    for row in rows:
        if abs(sum(item * item for item in row) - 1.0) > 1e-6:
            raise invalid("camera_to_table rotation is not orthonormal")
    if any(
        abs(sum(rows[i][k] * rows[j][k] for k in range(3))) > 1e-6
        for i, j in ((0, 1), (0, 2), (1, 2))
    ):
        raise invalid("camera_to_table rotation is not orthogonal")
    determinant = (
        rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7])
        - rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6])
        + rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6])
    )
    if abs(determinant - 1.0) > 1e-6:
        raise invalid("camera_to_table rotation must be right-handed")
    return matrix


def parse_camera_model(corpus: Mapping[str, object], resolution: tuple[int, int]) -> CameraModel:
    physical = mapping(corpus.get("physical_to_sim"), "physical_to_sim")
    _exact_fields(
        physical,
        {"direction", "matrix_2x3", "physical_units", "simulation_units"},
        "physical_to_sim",
    )
    if (
        physical["direction"] != "physical_table_to_simulation_table"
        or physical["physical_units"] != "meters"
        or physical["simulation_units"] != "meters"
    ):
        raise invalid("physical/simulation units or frame direction are invalid")
    try:
        transform = parse_rigid_se2(physical["matrix_2x3"])
        declared = parse_rigid_se2(corpus.get("physical_to_sim_se2"))
    except RolloutViolation as exc:
        raise invalid(f"physical_to_sim transform is invalid: {exc}") from exc
    if transform != declared:
        raise invalid("physical_to_sim transform declarations disagree")
    return CameraModel(
        _parse_intrinsics(corpus.get("intrinsics"), resolution),
        _parse_distortion(corpus.get("distortion")),
        _parse_extrinsics(corpus.get("camera_to_table")),
        transform,
    )


def _project(model: CameraModel, table: tuple[float, float, float]) -> tuple[float, float, float]:
    matrix = model.camera_to_table
    translated = (table[0] - matrix[3], table[1] - matrix[7], table[2] - matrix[11])
    # camera_to_table stores x_table = R_ct x_camera + t_ct. Projecting a
    # table point therefore requires the rigid inverse R_ct^T(x_table-t_ct).
    camera = tuple(
        sum(matrix[axis * 4 + row] * translated[axis] for axis in range(3)) for row in range(3)
    )
    x, y, depth = camera
    if depth <= 1e-6:
        raise invalid("table point is behind the camera")
    xn, yn = x / depth, y / depth
    k1, k2, p1, p2, k3 = model.distortion
    radius2 = xn * xn + yn * yn
    radial = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (radius2 + 2 * xn * xn)
    yd = yn * radial + p1 * (radius2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return (
        model.intrinsics[0] * xd + model.intrinsics[2],
        model.intrinsics[4] * yd + model.intrinsics[5],
        depth,
    )


def _nondegenerate(points: list[tuple[float, float]], label: str, minimum_area: float) -> None:
    if len(set(points)) < 4:
        raise invalid(f"{label} geometry lacks distinct points")
    origin = points[0]
    area = max(
        abs((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))
        for a in points[1:]
        for b in points[1:]
    )
    if area <= minimum_area:
        raise invalid(f"{label} geometry is degenerate")


def evaluate_correspondences(
    model: CameraModel,
    rows: list[dict[str, object]],
    *,
    label: str,
    resolution: tuple[int, int],
) -> tuple[float, float, float, float, float]:
    image_points: list[tuple[float, float]] = []
    table_xy: list[tuple[float, float]] = []
    projection_squared: list[float] = []
    simulation_squared: list[float] = []
    depths: list[float] = []
    r00, r01, tx, r10, r11, ty = model.physical_to_sim
    for row in rows:
        image = cast(
            "tuple[float, float]", vector(row.get("image_point_px"), f"{label}.image_point_px", 2)
        )
        table = cast(
            "tuple[float, float, float]",
            vector(row.get("table_point_m"), f"{label}.table_point_m", 3),
        )
        simulation = cast(
            "tuple[float, float]",
            vector(row.get("simulation_point_m"), f"{label}.simulation_point_m", 2),
        )
        if not (0 <= image[0] < resolution[0] and 0 <= image[1] < resolution[1]):
            raise invalid(f"{label} image point is outside the raw image")
        projected_x, projected_y, depth = _project(model, table)
        projection_squared.append((projected_x - image[0]) ** 2 + (projected_y - image[1]) ** 2)
        expected_sim = (r00 * table[0] + r01 * table[1] + tx, r10 * table[0] + r11 * table[1] + ty)
        simulation_squared.append(
            (expected_sim[0] - simulation[0]) ** 2 + (expected_sim[1] - simulation[1]) ** 2
        )
        image_points.append(image)
        table_xy.append((table[0], table[1]))
        depths.append(depth)
    _nondegenerate(image_points, f"{label} image", 1.0)
    _nondegenerate(table_xy, f"{label} table", 1e-6)
    projection_errors = [math.sqrt(item) for item in projection_squared]
    simulation_errors = [math.sqrt(item) for item in simulation_squared]
    return (
        math.sqrt(sum(projection_squared) / len(rows)),
        max(projection_errors),
        math.sqrt(sum(simulation_squared) / len(rows)),
        max(simulation_errors),
        max(depths),
    )
