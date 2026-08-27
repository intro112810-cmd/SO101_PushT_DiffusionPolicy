from __future__ import annotations

import pytest

from so101_pusht_benchmark.task.metric import coverage_fraction, success

SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def test_geometry_coverage_rotation_overlap_and_threshold() -> None:
    assert coverage_fraction(SQUARE, ((2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0))) == 0.0
    assert coverage_fraction(SQUARE, SQUARE) == 1.0
    assert coverage_fraction(SQUARE[::-1], SQUARE[::-1]) == 1.0
    assert 0.0 < coverage_fraction(SQUARE, ((0.5, -0.2), (1.2, 0.5), (0.5, 1.2), (-0.2, 0.5))) < 1.0
    assert success(0.95, threshold=0.95)
    assert not success(0.949999, threshold=0.95)
    assert success(max_coverage=0.97, final_coverage=0.95, threshold=0.95)
    with pytest.raises(ValueError, match="coverage"):
        success(1.1, threshold=0.95)
