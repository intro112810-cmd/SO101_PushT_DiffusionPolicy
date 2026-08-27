from __future__ import annotations

import numpy as np
import pytest

from so101_pusht_benchmark.control.action_filter import ActionFilter
from so101_pusht_benchmark.task.spec import APPROACH_HEIGHT_M, CONTACT_HEIGHT_M


@pytest.mark.parametrize("boundary", [CONTACT_HEIGHT_M, APPROACH_HEIGHT_M])
def test_float32_z_boundary_preserves_request_and_reports_canonicalization(boundary: float) -> None:
    action = np.asarray([0.25, 0.0, boundary], dtype=np.float32)
    original = action.copy()
    raw_z = float(action[2])
    action_filter = ActionFilter(1.0, (0.25, 0.0, boundary))

    result = action_filter.apply(action)

    assert result.requested == (0.25, 0.0, raw_z)
    assert result.applied == (0.25, 0.0, boundary)
    assert result.requested != result.applied
    assert result.clipped is True
    np.testing.assert_array_equal(action, original)


@pytest.mark.parametrize("boundary", [CONTACT_HEIGHT_M, APPROACH_HEIGHT_M])
def test_values_adjacent_to_float32_z_boundaries_are_not_canonicalized(boundary: float) -> None:
    encoded = np.float32(boundary)
    adjacent_values = (
        np.nextafter(encoded, np.float32(-np.inf)),
        np.nextafter(encoded, np.float32(np.inf)),
    )

    for adjacent in adjacent_values:
        raw_z = float(adjacent)
        action = np.asarray([0.25, 0.0, adjacent], dtype=np.float32)
        result = ActionFilter(1.0, (0.25, 0.0, raw_z)).apply(action)

        assert result.requested == (0.25, 0.0, raw_z)
        assert result.applied == result.requested
        assert result.clipped is False


def test_filter_does_not_mutate_input_view() -> None:
    backing = np.asarray([9.0, 0.25, 0.0, APPROACH_HEIGHT_M, 8.0], dtype=np.float32)
    original = backing.copy()
    action_view = backing[1:4]

    ActionFilter(1.0, (0.25, 0.0, APPROACH_HEIGHT_M)).apply(action_view)

    np.testing.assert_array_equal(backing, original)


@pytest.mark.parametrize(
    ("action", "initial", "max_step", "expected_applied"),
    [
        (
            (0.40, 0.0, 0.06),
            (0.38, 0.0, float(np.float32(0.06))),
            1.0,
            (0.38, 0.0, float(np.float32(0.06))),
        ),
        (
            (0.30, 0.0, 0.06),
            (0.25, 0.0, float(np.float32(0.06))),
            0.01,
            (0.26, 0.0, float(np.float32(0.06))),
        ),
    ],
)
def test_clamp_and_slew_preserve_requested_and_report_changed_applied(
    action: tuple[float, float, float],
    initial: tuple[float, float, float],
    max_step: float,
    expected_applied: tuple[float, float, float],
) -> None:
    encoded = np.asarray(action, dtype=np.float32)
    raw_request = (float(encoded[0]), float(encoded[1]), float(encoded[2]))

    result = ActionFilter(max_step, initial).apply(encoded)

    assert result.requested == raw_request
    assert result.applied == expected_applied
    assert result.requested != result.applied
    assert result.clipped is True
