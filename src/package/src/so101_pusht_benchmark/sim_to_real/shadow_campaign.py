"""Planner-complete, non-actuating continuous shadow orchestrator."""

from __future__ import annotations

import numpy as np

from so101_pusht_benchmark.sim_to_real.ledger_chain import GENESIS_DIGEST
from so101_pusht_benchmark.sim_to_real.ledger_io import LedgerDocument
from so101_pusht_benchmark.sim_to_real.physical_ik import build_physical_ik_planner
from so101_pusht_benchmark.sim_to_real.physical_ik_collision import pinned_model_digest
from so101_pusht_benchmark.sim_to_real.physical_ik_scene_pose import (
    ScenePoseExpectations,
    parse_scene_object_pose_receipt,
)
from so101_pusht_benchmark.sim_to_real.receipt_routing import prepare_receipt_directory
from so101_pusht_benchmark.sim_to_real.replay_history import (
    HistoryEvidence,
    build_history,
    build_receipt,
    validate_camera_receipt,
    validate_joint_receipt,
    validate_lineage_receipt,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.shadow_decision import (
    cartesian_decision,
    ik_decision,
    inference_decision,
    samples_decision,
    supervisor_acceptance,
    supervisor_rejection,
)
from so101_pusht_benchmark.sim_to_real.shadow_ledger import append_record, persist_campaign
from so101_pusht_benchmark.sim_to_real.shadow_samples import (
    load_campaign_scene_pose,
    load_campaign_samples,
    sample_age_seconds,
    samples_as_physical_samples,
)
from so101_pusht_benchmark.sim_to_real.shadow_types import (
    IKPlanner,
    ShadowCampaignInput,
    ShadowCampaignResult,
)
from so101_pusht_benchmark.sim_to_real.supervisor import (
    LINEAGE_DIGEST,
    RolloutSupervisor,
    SupervisorEvidence,
)
from so101_pusht_benchmark.sim_to_real.task_frame_bridge import (
    BridgeInput,
    build_task_frame_bridge,
    parse_mocap_xy,
)


class _CycleContext:
    """Mutable planner state shared only across sequential shadow cycles."""

    __slots__ = ("digests", "lineage", "planner", "supervisor")

    def __init__(
        self,
        lineage: LedgerDocument,
        digests: tuple[str, str],
        supervisor: RolloutSupervisor,
    ) -> None:
        self.lineage = lineage
        self.digests = digests
        self.supervisor = supervisor
        self.planner: IKPlanner | None = None


def _run_cycle(
    inputs: ShadowCampaignInput,
    context: _CycleContext,
    cycle: int,
    decisions: list[LedgerDocument],
) -> None:
    """Execute one read-only C1-C4 cycle or raise the exact rejection code."""
    samples = load_campaign_samples(inputs)
    decisions.append(
        samples_decision(
            cycle,
            samples,
            inputs.policy,
            fixture_only=bool(context.lineage["fixture_only"]),
        )
    )
    now = float(inputs.clock())
    if sample_age_seconds(samples, now) > inputs.policy.timing.sample_max_age_seconds:
        raise RolloutViolation(RolloutCode.R_STALE, "campaign samples are stale")
    joint_digest, camera_digest = context.digests
    history = build_history(
        HistoryEvidence(
            samples=samples,
            joint_document=inputs.joint_document,
            camera_document=inputs.camera_document,
            lineage_document=inputs.lineage_document,
            lineage_authority_digest=inputs.lineage_authority_digest,
            source_frame_path=inputs.source_frame_path,
        )
    )
    receipt = build_receipt(
        history,
        lineage=context.lineage,
        joint_digest=joint_digest,
        camera_digest=camera_digest,
        policy_seed=inputs.policy_seed,
    )
    decisions.append(inference_decision(cycle, receipt))
    mocap = parse_mocap_xy(np.asarray(receipt.action_chunk[0], dtype=np.float32))
    cartesian = build_task_frame_bridge(
        BridgeInput(
            camera_corpus=inputs.camera_corpus,
            policy=inputs.policy,
            raw_xy=mocap,
            previous_applied=None,
            ik=None,
        )
    )
    decisions.append(cartesian_decision(cycle, cartesian, inputs.camera_corpus))
    physical_samples = samples_as_physical_samples(samples)
    active_planner = context.planner if context.planner is not None else build_physical_ik_planner()
    second = physical_samples[-1]
    pose_document = load_campaign_scene_pose(inputs)
    pose_digest = pose_document.get("digest")
    if not isinstance(pose_digest, str):
        raise RolloutViolation(RolloutCode.R_MISSING, "campaign scene pose digest")
    scene_pose = parse_scene_object_pose_receipt(
        pose_document,
        inputs.policy,
        active_planner.collision_workspace,
        ScenePoseExpectations(
            pose_digest,
            second.record_id,
            second.created_at,
            second.digest,
            second.device_digest,
            camera_digest,
            pinned_model_digest(),
            second.created_at + 0.01,
        ),
    )
    proposal = active_planner.plan(
        target=cartesian,
        seed_degrees=physical_samples[-1].body_degrees,
        joint_equivalence_digest=joint_digest,
        policy=inputs.policy,
        scene_pose=scene_pose,
    )
    decisions.append(ik_decision(cycle, proposal))
    token = context.supervisor.mint(
        SupervisorEvidence(
            lineage_digest=LINEAGE_DIGEST,
            samples=physical_samples,
            joint_digest=joint_digest,
            camera_digest=camera_digest,
            policy=inputs.policy,
            cartesian=cartesian,
            ik_proposal=proposal,
            exclusive_owner=True,
            deadman_active=True,
            stop_clear=True,
            command_id=f"command-{cycle:03d}",
            command_budget=1,
        )
    )
    decisions.append(supervisor_acceptance(cycle, token))
    context.planner = active_planner


def run_shadow_campaign(inputs: ShadowCampaignInput) -> ShadowCampaignResult:
    """Run one planner-complete shadow campaign with zero physical writes."""
    if inputs.cycle_limit < 1:
        raise RolloutViolation(RolloutCode.R_BUDGET_EXHAUSTED, "campaign needs one cycle")
    lineage = validate_lineage_receipt(
        inputs.lineage_document,
        expected_digest=inputs.lineage_authority_digest,
    )
    fixture_only = bool(lineage["fixture_only"])
    output_dir = prepare_receipt_directory(inputs.output_dir, production=not fixture_only)
    ledger_path = output_dir / "ledger.jsonl"
    if fixture_only:
        digests = (
            validate_joint_receipt(inputs.joint_document),
            validate_camera_receipt(inputs.camera_document),
        )
    else:
        if inputs.production_receipt_digests is None:
            raise RolloutViolation(
                RolloutCode.R_POLICY_UNAUTHORIZED, "production receipt bindings missing"
            )
        expected_joint, expected_camera = inputs.production_receipt_digests
        digests = (
            validate_joint_receipt(inputs.joint_document, expected_digest=expected_joint),
            validate_camera_receipt(
                inputs.camera_document,
                expected_digest=expected_camera,
                expected_scope="authorized_physical_diagnostic",
            ),
        )
    context = _CycleContext(lineage, digests, RolloutSupervisor(inputs.clock))
    previous_digest = GENESIS_DIGEST
    records: list[LedgerDocument] = []
    completed_cycles = 0
    hold_code: str | None = None
    for cycle in range(inputs.cycle_limit):
        decisions: list[LedgerDocument] = []
        try:
            _run_cycle(inputs, context, cycle, decisions)
        except RolloutViolation as exc:
            hold_code = exc.code.value
            after_stage = str(decisions[-1]["kind"]) if decisions else "none"
            decisions.append(supervisor_rejection(cycle, exc, after_stage=after_stage))
        for decision in decisions:
            previous_digest = append_record(
                records,
                decision,
                previous_digest=previous_digest,
            )
        if hold_code is not None:
            previous_digest = append_record(
                records,
                {
                    "kind": "hold",
                    "cycle": cycle,
                    "terminal_state": "HOLD",
                    "terminal_code": hold_code,
                },
                previous_digest=previous_digest,
            )
            break
        completed_cycles += 1
    terminal_state = "SHADOW_COMPLETE" if hold_code is None else "HOLD"
    if terminal_state == "SHADOW_COMPLETE":
        previous_digest = append_record(
            records,
            {"kind": "campaign_complete", "terminal_state": terminal_state},
            previous_digest=previous_digest,
        )
    previous_digest = append_record(
        records,
        {
            "kind": "cleanup",
            "status": "released",
            "terminal_state": terminal_state,
            "writer_closed": True,
            "motor_writes_performed": False,
            "actuation_performed": False,
            "writer_symbols": 0,
            "read_only": True,
        },
        previous_digest=previous_digest,
    )
    result = ShadowCampaignResult(
        terminal_state=terminal_state,
        terminal_code=hold_code if hold_code is not None else terminal_state,
        cycles_completed=completed_cycles if terminal_state == "SHADOW_COMPLETE" else 0,
        cycle_limit=inputs.cycle_limit,
        ledger_digest=previous_digest,
        motor_writes_performed=False,
        actuation_performed=False,
        writer_symbols=0,
        evidence_scope="test_fixture_only" if fixture_only else "production",
        policy_evidence=(
            "fixture_adapter_not_frozen_production"
            if fixture_only
            else "authentic_frozen_production"
        ),
        receipt_path=output_dir / "SHADOW_COMPLETE",
        ledger_path=ledger_path,
    )
    persist_campaign(records, result, output_dir)
    return result
