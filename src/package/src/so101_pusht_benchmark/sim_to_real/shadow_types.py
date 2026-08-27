"""Typed inputs, results, and seams for the continuous shadow orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol

from so101_pusht_benchmark.sim_to_real.physical_ik import PhysicalIKProposal
from so101_pusht_benchmark.sim_to_real.physical_ik_fk import MuJoCoWorkspace
from so101_pusht_benchmark.sim_to_real.physical_ik_scene_pose import SceneObjectPoseReceipt
from so101_pusht_benchmark.sim_to_real.policy_types import (
    FixtureApprovedSafetyPolicy,
    ProductionApprovedSafetyPolicy,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.task_frame_bridge import CartesianProposalReceipt

__all__ = (
    "CampaignClock",
    "FixtureClock",
    "IKPlanner",
    "ShadowCampaignInput",
    "ShadowCampaignResult",
)


class IKPlanner(Protocol):
    """Narrow physical-IK seam; it is never a writer."""

    @property
    def collision_workspace(self) -> MuJoCoWorkspace: ...

    def plan(
        self,
        *,
        target: CartesianProposalReceipt,
        seed_degrees: object,
        joint_equivalence_digest: str,
        policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
        scene_pose: SceneObjectPoseReceipt | None = None,
    ) -> PhysicalIKProposal: ...


class CampaignClock(Protocol):
    """Deterministic monotonic campaign clock; production uses ``time.monotonic``."""

    def __call__(self) -> float: ...


@dataclass(frozen=True, slots=True)
class FixtureClock:
    """Fake advancing clock with no wall-clock or sleep dependence."""

    start: float
    step: float
    _current: float | None = None

    def __call__(self) -> float:
        value = self.start if self._current is None else self._current + self.step
        if not math.isfinite(value):
            raise RolloutViolation(RolloutCode.R_NONFINITE, "campaign clock")
        object.__setattr__(self, "_current", value)
        return value


@dataclass(frozen=True, slots=True)
class ShadowCampaignInput:
    """All provenance and policy inputs for one shadow campaign."""

    fixture_dir: Path
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy
    lineage_document: dict[str, object]
    lineage_authority_digest: str
    joint_document: dict[str, object]
    camera_document: dict[str, object]
    camera_corpus: dict[str, object]
    source_frame_path: Path
    output_dir: Path
    clock: CampaignClock
    cycle_limit: int
    policy_seed: int
    production_receipt_digests: tuple[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ShadowCampaignResult:
    """Non-actuating terminal outcome plus the campaign evidence digest."""

    terminal_state: str
    terminal_code: str
    cycles_completed: int
    cycle_limit: int
    ledger_digest: str
    motor_writes_performed: bool
    actuation_performed: bool
    writer_symbols: int
    evidence_scope: str
    policy_evidence: str
    receipt_path: Path
    ledger_path: Path

    def to_document(self) -> dict[str, object]:
        """Encode the machine-consumed campaign receipt."""
        return {
            "schema": 1,
            "mode": "sim_to_real_continuous_shadow_campaign",
            "terminal_state": self.terminal_state,
            "terminal_code": self.terminal_code,
            "cycles_completed": self.cycles_completed,
            "cycle_limit": self.cycle_limit,
            "ledger_digest": self.ledger_digest,
            "motor_writes_performed": self.motor_writes_performed,
            "actuation_performed": self.actuation_performed,
            "writer_symbols": self.writer_symbols,
            "evidence_scope": self.evidence_scope,
            "policy_evidence": self.policy_evidence,
        }
