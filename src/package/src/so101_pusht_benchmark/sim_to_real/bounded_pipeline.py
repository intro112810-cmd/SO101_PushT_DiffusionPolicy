"""One bounded cycle through the corrected shadow planning seams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np

from .authorization import AuthorizationToken
from .ledger_chain import canonical_hash
from .physical_ik import PhysicalIKProposal, build_physical_ik_planner
from .physical_ik_collision import pinned_model_digest
from .physical_ik_scene_pose import ScenePoseExpectations, parse_scene_object_pose_receipt
from .policy_types import FixtureApprovedSafetyPolicy
from .replay_history import (
    HistoryEvidence,
    build_history,
    build_receipt,
    validate_camera_receipt,
    validate_joint_receipt,
    validate_lineage_receipt,
)
from .rollout_codes import RolloutCode, RolloutViolation
from .rollout_record_types import PhysicalSample
from .shadow_decision import (
    cartesian_decision,
    ik_decision,
    inference_decision,
    samples_decision,
    supervisor_acceptance,
)
from .shadow_samples import samples_as_physical_samples
from .shadow_types import CampaignClock
from .supervisor import LINEAGE_DIGEST, RolloutSupervisor, SupervisorEvidence
from .task_frame_bridge import BridgeInput, build_task_frame_bridge, parse_mocap_xy

_ROOT: Final = Path(__file__).resolve().parents[3]
_FIXTURES: Final = _ROOT / "tests/fixtures/sim_to_real"
_LINEAGE_AUTHORITY: Final = "192d568795b756ac1edcde78a4a24ed8d37f1fef3bde14cd32a6d441c221a5e4"


@dataclass(frozen=True, slots=True)
class PlannedCycle:
    records: tuple[dict[str, object], ...]
    samples: tuple[PhysicalSample, PhysicalSample]
    proposal: PhysicalIKProposal
    token: AuthorizationToken
    inference_digest: str
    action_digest: str


def _json(name: str) -> dict[str, object]:
    import json
    from typing import cast

    raw = json.loads((_FIXTURES / name).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, f"fixture {name} must be a mapping")
    return cast("dict[str, object]", raw)


@dataclass(frozen=True, slots=True)
class CyclePlanningInput:
    cycle: int
    sample_records: tuple[dict[str, object], dict[str, object]]
    policy: FixtureApprovedSafetyPolicy
    clock: CampaignClock
    command_id: str
    policy_seed: int
    scene_pose_document: dict[str, object]


def plan_fixture_cycle(inputs: CyclePlanningInput) -> PlannedCycle:
    """Observe, infer eight actions, select zero, transform, plan, and authorize."""
    cycle = inputs.cycle
    sample_records = inputs.sample_records
    policy = inputs.policy
    lineage = _json("lineage.json")
    joint = _json("joint-equivalence.json")
    camera = _json("camera-registration.json")
    corpus = _json("camera_registration_valid/corpus.json")
    lineage_typed = validate_lineage_receipt(lineage, expected_digest=_LINEAGE_AUTHORITY)
    joint_digest = validate_joint_receipt(joint)
    camera_digest = validate_camera_receipt(camera)
    sample_decision = samples_decision(
        cycle, sample_records, policy, fixture_only=bool(lineage_typed["fixture_only"])
    )
    sample_decision["scene_pose_receipt"] = dict(inputs.scene_pose_document)
    records: list[dict[str, object]] = [sample_decision]
    history = build_history(
        HistoryEvidence(
            samples=sample_records,
            joint_document=joint,
            camera_document=camera,
            lineage_document=lineage,
            lineage_authority_digest=_LINEAGE_AUTHORITY,
            source_frame_path=_FIXTURES / "physical_frame.png",
        )
    )
    receipt = build_receipt(
        history,
        lineage=lineage_typed,
        joint_digest=joint_digest,
        camera_digest=camera_digest,
        policy_seed=inputs.policy_seed,
    )
    inference = inference_decision(cycle, receipt)
    records.append(inference)
    mocap = parse_mocap_xy(np.asarray(receipt.action_chunk[0], dtype=np.float32))
    cartesian = build_task_frame_bridge(BridgeInput(corpus, policy, mocap, None, None))
    records.append(cartesian_decision(cycle, cartesian, corpus))
    samples = samples_as_physical_samples(sample_records)
    planner = build_physical_ik_planner()
    second = sample_records[-1]
    pose_digest = inputs.scene_pose_document.get("digest")
    if not isinstance(pose_digest, str):
        raise RolloutViolation(RolloutCode.R_MISSING, "scene pose digest")
    scene_pose = parse_scene_object_pose_receipt(
        inputs.scene_pose_document,
        policy,
        planner.collision_workspace,
        ScenePoseExpectations(
            pose_digest,
            str(second["record_id"]),
            float(cast("float", second["created_at"])),
            str(second["digest"]),
            str(second["device_digest"]),
            camera_digest,
            pinned_model_digest(),
            float(cast("float", second["created_at"])) + 0.01,
        ),
    )
    proposal = planner.plan(
        target=cartesian,
        seed_degrees=samples[-1].body_degrees,
        joint_equivalence_digest=joint_digest,
        policy=policy,
        scene_pose=scene_pose,
    )
    records.append(ik_decision(cycle, proposal))
    supervisor = RolloutSupervisor(inputs.clock)
    token = supervisor.mint(
        SupervisorEvidence(
            LINEAGE_DIGEST,
            samples,
            joint_digest,
            camera_digest,
            policy,
            cartesian,
            proposal,
            True,
            True,
            True,
            inputs.command_id,
            1,
        )
    )
    records.append(supervisor_acceptance(cycle, token))
    inference_digest = str(inference["inference_digest"])
    return PlannedCycle(
        tuple(records),
        samples,
        proposal,
        token,
        inference_digest,
        canonical_hash({"action_chunk": receipt.action_chunk_float32_2d}),
    )
