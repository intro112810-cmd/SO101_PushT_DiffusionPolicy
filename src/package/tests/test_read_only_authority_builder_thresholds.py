"""No-default signing boundary for read-only acquisition thresholds."""

from __future__ import annotations

import sys

import pytest

from so101_pusht_benchmark.sim_to_real.read_only_authority_builder import main
from so101_pusht_benchmark.sim_to_real.read_only_authority_thresholds import (
    AcquisitionThresholdError,
    AcquisitionThresholdInputs,
)


def test_builder_cli_requires_every_signed_timeout_without_opening_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build-read-only-authority",
            "--profile",
            "/must-not-open/profile.yaml",
            "--output-dir",
            "/must-not-create/output",
            "--source-lineage-authority-digest",
            "a" * 64,
        ],
    )

    with pytest.raises(SystemExit) as caught:
        main()

    assert caught.value.code == 2


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_timeout_rejects_before_signing(invalid: float) -> None:
    with pytest.raises(AcquisitionThresholdError):
        AcquisitionThresholdInputs(
            invalid,
            5.0,
            0.2,
            1.0,
            1,
            2,
            0.2,
            0.04,
            0.003,
            1.5,
            2.0,
            12,
        )


@pytest.mark.parametrize(("priming", "pairs"), [(0, 2), (2, 2), (1, 1), (1, 3)])
def test_non_exact_capture_cardinality_rejects_before_signing(
    priming: int,
    pairs: int,
) -> None:
    with pytest.raises(AcquisitionThresholdError):
        AcquisitionThresholdInputs(
            5.0,
            5.0,
            0.2,
            1.0,
            priming,
            pairs,
            0.2,
            0.04,
            0.003,
            1.5,
            2.0,
            12,
        )
