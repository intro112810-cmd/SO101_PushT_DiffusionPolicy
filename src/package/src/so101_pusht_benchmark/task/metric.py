"""Pure geometry metrics for target coverage and success."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["coverage_fraction", "success"]

Point = tuple[float, float]
Polygon = Sequence[Point]


def _signed_area(polygon: Polygon) -> float:
    if not polygon:
        return 0.0
    return (
        sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(polygon, (*polygon[1:], polygon[0]), strict=True)
        )
        / 2
    )


def _area(polygon: Polygon) -> float:
    return abs(_signed_area(polygon))


def _cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _intersection(start: Point, end: Point, edge_start: Point, edge_end: Point) -> Point:
    dx, dy = end[0] - start[0], end[1] - start[1]
    ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]
    denominator = dx * ey - dy * ex
    if denominator == 0:
        return end
    t = ((edge_start[0] - start[0]) * ey - (edge_start[1] - start[1]) * ex) / denominator
    return (start[0] + t * dx, start[1] + t * dy)


def _clip(subject: list[Point], edge_start: Point, edge_end: Point) -> list[Point]:
    if not subject:
        return []
    result: list[Point] = []
    previous = subject[-1]
    previous_inside = _cross(edge_start, edge_end, previous) >= 0
    for current in subject:
        current_inside = _cross(edge_start, edge_end, current) >= 0
        if current_inside != previous_inside:
            result.append(_intersection(previous, current, edge_start, edge_end))
        if current_inside:
            result.append(current)
        previous, previous_inside = current, current_inside
    return result


def coverage_fraction(target: Polygon, placed: Polygon) -> float:
    """Return the area of target covered by placed, independent of winding."""
    target_area = _area(target)
    if target_area <= 0 or _area(placed) <= 0:
        raise ValueError("polygons must have positive area")
    clip_polygon = list(target) if _signed_area(target) > 0 else list(reversed(target))
    clipped = list(placed)
    for edge_start, edge_end in zip(
        clip_polygon, (*clip_polygon[1:], clip_polygon[0]), strict=True
    ):
        clipped = _clip(clipped, edge_start, edge_end)
    return max(0.0, min(1.0, _area(clipped) / target_area))


def success(
    max_coverage: float, final_coverage: float | None = None, threshold: float = 0.95
) -> bool:
    """Apply the versioned maximum and final coverage acceptance rule."""
    values = (
        (max_coverage, threshold)
        if final_coverage is None
        else (max_coverage, final_coverage, threshold)
    )
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("coverage and threshold must be in [0, 1]")
    return max_coverage >= threshold and (final_coverage is None or final_coverage >= threshold)
