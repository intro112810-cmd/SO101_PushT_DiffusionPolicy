from __future__ import annotations

import pytest

from so101_pusht_benchmark.core.contract import ContractError, TimingContract


def test_timing_is_pre_action_10hz_and_50_substeps() -> None:
    timing = TimingContract.create(frame_index=3, timestamp=0.3)
    assert timing.substeps == 50
    assert timing.action_interval == (0.3, 0.4)
    with pytest.raises(ContractError):
        TimingContract.create(frame_index=3, timestamp=0.31)


def test_timing_sequence_rejects_off_by_one_or_discontinuity() -> None:
    first = TimingContract.create(0, 0.0)
    second = TimingContract.create(1, 0.1)
    first.validate_next(second)
    with pytest.raises(ContractError):
        first.validate_next(TimingContract.create(2, 0.2))
    with pytest.raises(ContractError):
        TimingContract.create(1, 0.2)
