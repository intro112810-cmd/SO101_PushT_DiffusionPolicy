from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from so101_pusht_benchmark.sim_to_real.affine_state import (
    load_joint_mapping_receipt,
    mapped_agent_pos_from_receipt,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

BENCHMARK = Path(__file__).resolve().parents[1]
VALID = BENCHMARK / "tests/fixtures/sim_to_real/joint_map_valid.json"
INVALID = Path(
    "/home/intro/InternLab/02_InTro_Project/04_experiments/so101_pusht_benchmark/"
    "inference/joint_mapping_calibration_20260823/joint_mapping_receipt.json"
)


def test_valid_mapping_receipt_yields_float32_five() -> None:
    receipt = load_joint_mapping_receipt(VALID)
    agent_pos, source, evidence = mapped_agent_pos_from_receipt(receipt, receipt_path=VALID)
    assert agent_pos.dtype == np.float32
    assert agent_pos.shape == (5,)
    assert source == "receipt_bound_affine_mapping"
    assert evidence["joint_map_receipt_sha256"]


def test_invalid_elbow_receipt_raises_before_policy() -> None:
    receipt = load_joint_mapping_receipt(INVALID)
    with pytest.raises(RolloutViolation) as exc_info:
        mapped_agent_pos_from_receipt(receipt, receipt_path=INVALID)
    assert exc_info.value.code is RolloutCode.R_INVALID_ELBOW
    assert "elbow_flex" in str(exc_info.value)
