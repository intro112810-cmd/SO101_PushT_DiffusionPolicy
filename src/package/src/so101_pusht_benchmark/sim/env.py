"""Deterministic, fail-closed 10 Hz SO-101 Push-T MuJoCo environment."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray
from ..control.action_filter import ActionFilter
from ..task.metric import coverage_fraction
from ..task.spec import CONTACT_HEIGHT_M
from .dls_ik import DLSIK
from .safety import Fault, SafetyState, allowed_contact
from .scene import Scene


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: dict[str, NDArray[np.generic]]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class PushTEnv:
    def __init__(self, development: bool = False) -> None:
        self.scene = Scene()
        self.development = development
        self.safety = SafetyState()
        self.filter = ActionFilter(0.015)
        self.ik = DLSIK(self.scene)
        self.frame = 0
        self.active = False
        self.max_coverage = 0.0
        self.ids = {
            name: self.scene.mujoco.mj_name2id(
                self.scene.model, self.scene.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            for name in ("push_t_x", "push_t_y", "push_t_yaw")
        }
        self.pusher = self.scene.mujoco.mj_name2id(
            self.scene.model, self.scene.mujoco.mjtObj.mjOBJ_BODY, "pusher"
        )

    def close(self) -> None:
        self.scene.close()

    @property
    def paper_state(self) -> tuple[float, float, float, float, float]:
        """T pose (x, y, yaw) then pusher XY, in task metres.

        Feeds the canonical PushT-style 2D paper view.
        """
        d = self.scene.data
        tx = float(d.qpos[self.ids["push_t_x"]])
        ty = float(d.qpos[self.ids["push_t_y"]])
        tyaw = float(d.qpos[self.ids["push_t_yaw"]])
        px, py, _ = d.xpos[self.pusher]
        return tx, ty, tyaw, float(px), float(py)

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, object]]:
        rng = np.random.default_rng(seed)
        m, d, mj = self.scene.model, self.scene.data, self.scene.mujoco
        mj.mj_resetData(m, d)
        self.safety.reset()
        self.frame = 0
        self.active = False
        for _ in range(100):
            x = float(rng.uniform(0.22, 0.30))
            y = float(rng.uniform(-0.10, 0.10))
            ex = float(rng.uniform(0.18, 0.38))
            ey = float(rng.uniform(-0.16, 0.16))
            if math.hypot(x - ex, y - ey) >= 0.08:
                break
        else:
            self.safety.latch(Fault.RESET_EXHAUSTED)
            raise RuntimeError("reset attempts exhausted")
        d.qpos[self.ids["push_t_x"]] = x
        d.qpos[self.ids["push_t_y"]] = y
        d.qpos[self.ids["push_t_yaw"]] = (
            0.0 if self.development else float(rng.uniform(-math.pi / 2, math.pi / 2))
        )
        d.mocap_pos[0] = (ex, ey, CONTACT_HEIGHT_M)
        mj.mj_forward(m, d)
        measured = d.xpos[self.pusher].copy()
        self.filter = ActionFilter(
            0.015, (float(measured[0]), float(measured[1]), float(measured[2]))
        )
        self.active = True
        return self.observe(), {"seed": seed, "fault": None}

    def observe(self) -> dict[str, NDArray[np.generic]]:
        return {
            "observation.images.front": self.scene.render(),
            "observation.state": np.concatenate(
                (
                    self.scene.data.qpos[:6],
                    self.scene.data.qvel[:6],
                    self.scene.data.xpos[self.pusher],
                )
            ).astype(np.float32),
        }

    def _coverage(self) -> float:
        x, y, yaw = (
            float(self.scene.data.qpos[self.ids[n]]) for n in ("push_t_x", "push_t_y", "push_t_yaw")
        )
        c, s = math.cos(yaw), math.sin(yaw)
        local = (
            (-0.055, -0.014),
            (0.055, -0.014),
            (0.055, 0.014),
            (0.014, 0.014),
            (0.014, 0.072),
            (-0.014, 0.072),
            (-0.014, 0.014),
            (-0.055, 0.014),
        )
        placed = [(x + c * a - s * b, y + s * a + c * b) for a, b in local]
        target = [
            (0.285, -0.014),
            (0.395, -0.014),
            (0.395, 0.014),
            (0.354, 0.014),
            (0.354, 0.072),
            (0.326, 0.072),
            (0.326, 0.014),
            (0.285, 0.014),
        ]
        return coverage_fraction(target, placed)

    def abort_collection(self, reason: str) -> None:
        """Revoke simulation control immediately; this never invokes physical E-stop hardware."""
        self.safety.latch(Fault.COLLECTION_ABORT)
        self._revoke_control(reason)
        self.collection_abort_reason = reason

    def stop_collection(self, reason: str) -> None:
        """Revoke control without latching safety for an operator terminal transition."""
        self._revoke_control(reason)

    def _revoke_control(self, reason: str) -> None:
        d = self.scene.data
        # Abort must hold the target at the measured pusher pose, not at the
        # last requested (possibly distant) mocap target.
        measured = d.xpos[self.pusher].copy()
        d.mocap_pos[0] = measured
        self.filter = ActionFilter(
            0.015, (float(measured[0]), float(measured[1]), float(measured[2]))
        )
        d.ctrl[:] = d.qpos[: d.ctrl.size]
        self.active = False
        self.terminal_reason = reason

    def _fault(self, reason: Fault) -> StepResult:
        self.safety.latch(reason)
        self._revoke_control(reason.value)
        return StepResult(self.observe(), 0.0, True, False, {"fault": reason.value})

    def step(self, action: object) -> StepResult:
        if not self.active or not self.safety.safe:
            return self._fault(Fault.TERMINAL)
        try:
            filtered = self.filter.apply(action)
        except ValueError:
            return self._fault(Fault.INVALID_ACTION)
        target = filtered.applied
        ik = self.ik.solve(target)
        if not ik.valid:
            return self._fault(Fault.INVALID_IK)
        d, m, mj = self.scene.data, self.scene.model, self.scene.mujoco
        d.mocap_pos[0] = target
        d.ctrl[:6] = np.clip(ik.qpos[:6], m.actuator_ctrlrange[:6, 0], m.actuator_ctrlrange[:6, 1])
        for _ in range(50):
            mj.mj_step(m, d)
            if not bool(np.isfinite(d.qpos).all()):
                return self._fault(Fault.NONFINITE_PHYSICS)
            for index, contact in enumerate(d.contact[: d.ncon]):
                a = m.geom(contact.geom1).name
                b = m.geom(contact.geom2).name
                if not allowed_contact(a or "", b or ""):
                    return self._fault(Fault.FORBIDDEN_CONTACT)
                force = np.zeros(6, dtype=np.float64)
                mj.mj_contactForce(m, d, index, force)
                if float(np.linalg.norm(force[:3])) > 40.0:
                    return self._fault(Fault.FORBIDDEN_CONTACT)
        self.frame += 1
        coverage = self._coverage()
        self.max_coverage = max(self.max_coverage, coverage)
        done = self.frame >= 300 or coverage >= 0.95
        if done:
            self.active = False
        return StepResult(
            self.observe(),
            float(coverage >= 0.95),
            done,
            self.frame >= 300,
            {
                "coverage": coverage,
                "max_coverage": self.max_coverage,
                "requested_target": filtered.requested,
                "applied_target": filtered.applied,
                "ik_residual": ik.residual,
                "ik_joint_target": ik.qpos[:6].astype(np.float32).tolist(),
                "ctrl_command": d.ctrl[:6].astype(np.float32).tolist(),
                "ack_status": "applied",
                "contact": d.ncon > 0,
                "dropped": False,
                "duplicate": False,
                "clipped": filtered.clipped,
                "timestamp": self.frame / 10,
            },
        )
