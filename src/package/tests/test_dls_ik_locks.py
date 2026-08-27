"""DLSIK must lock wrist_roll and gripper to fixed values.

The push task needs a spherical pusher whose axial rotation is irrelevant;
wrist_roll (joint 4) and gripper (joint 5) must stay frozen while the IK
solves shoulder_pan/shoulder_lift/elbow_flex/wrist_flex for the EE target.
"""

from __future__ import annotations

import numpy as np
import pytest

from so101_pusht_benchmark.sim.dls_ik import DLSIK
from so101_pusht_benchmark.sim.env import PushTEnv

LOCK_WRIST_ROLL = 0.0
LOCK_GRIPPER = 0.0


def _locked_arm_env() -> tuple[PushTEnv, DLSIK]:
    env = PushTEnv()
    env.reset(seed=0)
    ik = DLSIK(env.scene)
    return env, ik


@pytest.mark.parametrize(
    "target",
    [
        (0.28, 0.00, 0.045),
        (0.30, -0.05, 0.050),
        (0.25, 0.05, 0.045),
        (0.34, 0.00, 0.060),
        (0.20, 0.10, 0.045),
        (0.36, -0.08, 0.065),
    ],
)
def test_ik_keeps_wrist_roll_and_gripper_locked(target: tuple[float, float, float]) -> None:
    env, ik = _locked_arm_env()
    try:
        result = ik.solve(target)
        assert result.valid, f"target {target} must be reachable"
        assert result.residual < 0.005, f"target {target} residual {result.residual:.4f}"
        assert not np.allclose(  # type: ignore[reportUnknownMemberType]
            result.qpos[:4], 0.0, atol=1e-3
        ), f"returned qpos is the initial pose, not the converged solution (target {target})"
        assert float(result.qpos[4]) == LOCK_WRIST_ROLL, (
            f"wrist_roll drifted to {result.qpos[4]:.4f} for target {target}"
        )
        assert float(result.qpos[5]) == LOCK_GRIPPER, (
            f"gripper drifted to {result.qpos[5]:.4f} for target {target}"
        )
    finally:
        env.close()


def test_ik_restores_original_qpos_on_failure() -> None:
    env, ik = _locked_arm_env()
    try:
        original = env.scene.data.qpos.copy()  # type: ignore[reportUnknownMemberType]
        # Unreachable target far outside the workspace.
        ik.solve((0.9, 0.9, 0.5))
        assert np.array_equal(  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            env.scene.data.qpos,
            np.asarray(original),  # type: ignore[reportUnknownArgumentType]
        ), "failed solve must restore qpos"
    finally:
        env.close()
