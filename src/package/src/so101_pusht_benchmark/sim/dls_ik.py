"""Deterministic damped-least-squares Cartesian IK for the SO-101 arm.

Four DOF solve: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex. The
wrist_roll and gripper joints are frozen at fixed values because the pusher
is a sphere whose axial rotation does not affect the contact point; keeping
them locked also stops the arm from taking contorted poses while tracking.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .scene import Scene


@dataclass(frozen=True, slots=True)
class IKResult:
    target: tuple[float, float, float]
    qpos: NDArray[np.float64]
    residual: float
    iterations: int
    valid: bool


LOCK_WRIST_ROLL = 0.0
LOCK_GRIPPER = 0.0
FREE_JOINTS = 4  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex


class DLSIK:
    def __init__(self, scene: Scene, tolerance: float = 0.003, iterations: int = 64) -> None:
        self.scene = scene
        self.tolerance = tolerance
        self.iterations = iterations
        self.site_id = scene.mujoco.mj_name2id(
            scene.model, scene.mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )

    def solve(self, target: tuple[float, float, float]) -> IKResult:
        m, d, mj = self.scene.model, self.scene.data, self.scene.mujoco
        start: NDArray[np.float64] = d.qpos.copy()  # type: ignore[reportUnknownMemberType]
        jac: NDArray[np.float64] = np.empty((3, m.nv))
        rot: NDArray[np.float64] = np.empty((3, m.nv))
        residual = float("inf")
        for count in range(1, self.iterations + 1):
            error: NDArray[np.float64] = (
                np.asarray(target, dtype=np.float64) - d.site_xpos[self.site_id]
            )
            residual = float(np.linalg.norm(error))
            if residual <= self.tolerance:
                solved: NDArray[np.float64] = d.qpos[:6].copy()  # type: ignore[reportUnknownMemberType]
                return IKResult(target, solved, residual, count, True)
            mj.mj_jacSite(m, d, jac, rot, self.site_id)
            solved = np.asarray(
                np.linalg.solve(
                    jac[:, :FREE_JOINTS] @ jac[:, :FREE_JOINTS].T + 0.001 * np.eye(3), error
                ),
                dtype=np.float64,
            )
            delta: NDArray[np.float64] = jac[:, :FREE_JOINTS].T @ solved
            d.qpos[:FREE_JOINTS] = np.clip(
                d.qpos[:FREE_JOINTS] + delta,
                m.jnt_range[:FREE_JOINTS, 0] + 0.05,
                m.jnt_range[:FREE_JOINTS, 1] - 0.05,
            )
            d.qpos[FREE_JOINTS] = LOCK_WRIST_ROLL
            d.qpos[FREE_JOINTS + 1] = LOCK_GRIPPER
            mj.mj_forward(m, d)
        d.qpos[:] = start  # type: ignore[reportIndexOperator]
        mj.mj_forward(m, d)
        return IKResult(target, start[:6], residual, self.iterations, False)  # type: ignore[reportIndexOperator]
