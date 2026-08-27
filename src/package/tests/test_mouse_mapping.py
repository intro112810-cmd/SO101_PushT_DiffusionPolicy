import math

import pytest

from so101_pusht_benchmark.control.mouse_mapping import (
    pixels_to_task_xy,
    task_xy_to_pixels,
    MappingError,
)

BOUNDS_X = (0.18, 0.38)
BOUNDS_Y = (-0.16, 0.16)


def test_corners_mapping() -> None:
    pane = (100.0, 100.0)
    # Letterbox math for a 100x100 pane over a 0.20 x 0.32 m task area:
    # scale 312.5, active 62.5 x 100 px, offsets (18.75, 0).
    res = pixels_to_task_xy(18.75, 0.0, pane[0], pane[1], BOUNDS_X, BOUNDS_Y)
    assert math.isclose(res.x, BOUNDS_X[0], abs_tol=1e-5)
    assert math.isclose(res.y, BOUNDS_Y[1], abs_tol=1e-5)
    assert not res.outside

    # Bottom-Right of active area: px=18.75+62.5=81.25, py=100
    # Should map to max X, min Y
    res = pixels_to_task_xy(81.25, 100.0, pane[0], pane[1], BOUNDS_X, BOUNDS_Y)
    assert math.isclose(res.x, BOUNDS_X[1], abs_tol=1e-5)
    assert math.isclose(res.y, BOUNDS_Y[0], abs_tol=1e-5)
    assert not res.outside


def test_center_mapping() -> None:
    pane = (200.0, 200.0)
    res = pixels_to_task_xy(100.0, 100.0, pane[0], pane[1], BOUNDS_X, BOUNDS_Y)
    center_x = (BOUNDS_X[0] + BOUNDS_X[1]) / 2.0
    center_y = (BOUNDS_Y[0] + BOUNDS_Y[1]) / 2.0
    assert math.isclose(res.x, center_x, abs_tol=1e-5)
    assert math.isclose(res.y, center_y, abs_tol=1e-5)
    assert not res.outside


def test_letterbox_outside() -> None:
    pane = (100.0, 100.0)
    # (0, 0) is in the left letterbox margin (since active offset_x=18.75)
    res = pixels_to_task_xy(0.0, 50.0, pane[0], pane[1], BOUNDS_X, BOUNDS_Y)
    assert res.outside
    # It should be clamped to the boundary
    assert math.isclose(res.x, BOUNDS_X[0], abs_tol=1e-5)


def test_aspect_preservation_non_square() -> None:
    # task_w/task_h = 0.2 / 0.32 = 0.625
    # Let's make a pane that matches this exactly
    pane = (625.0, 1000.0)
    # px=625, py=1000 -> max X, min Y
    res = pixels_to_task_xy(625.0, 1000.0, pane[0], pane[1], BOUNDS_X, BOUNDS_Y)
    assert math.isclose(res.x, BOUNDS_X[1], abs_tol=1e-5)
    assert math.isclose(res.y, BOUNDS_Y[0], abs_tol=1e-5)
    assert not res.outside


def test_finite_value_rejection() -> None:
    pane = (100.0, 100.0)
    with pytest.raises(MappingError):
        pixels_to_task_xy(float("nan"), 50.0, pane[0], pane[1], BOUNDS_X, BOUNDS_Y)
    with pytest.raises(MappingError):
        pixels_to_task_xy(50.0, float("inf"), pane[0], pane[1], BOUNDS_X, BOUNDS_Y)


def test_round_trip() -> None:
    pane = (100.0, 100.0)
    task_x = 0.25
    task_y = 0.05

    px, py = task_xy_to_pixels(task_x, task_y, pane[0], pane[1], BOUNDS_X, BOUNDS_Y)
    res = pixels_to_task_xy(px, py, pane[0], pane[1], BOUNDS_X, BOUNDS_Y)

    assert not res.outside
    assert math.isclose(res.x, task_x, abs_tol=1e-5)
    assert math.isclose(res.y, task_y, abs_tol=1e-5)
