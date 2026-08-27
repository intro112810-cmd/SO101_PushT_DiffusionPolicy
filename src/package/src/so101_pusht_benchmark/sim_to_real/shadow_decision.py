"""Durable planner-complete shadow decisions and semantic replay checks."""

from __future__ import annotations

from collections.abc import Mapping

from .authorization import AuthorizationToken
from .ledger_chain import canonical_hash
from .physical_ik import PhysicalIKProposal
from .policy_types import FixtureApprovedSafetyPolicy, ProductionApprovedSafetyPolicy
from .replay_history import receipt_digest
from .replay_types import InferenceReceipt
from .rollout_codes import RolloutViolation
from .task_frame_bridge import CartesianProposalReceipt

LedgerDocument = dict[str, object]
_FIXTURE_POLICY_EVIDENCE = "fixture_adapter_not_frozen_production"
_FROZEN_POLICY_EVIDENCE = "authentic_frozen_production"


def samples_decision(
    cycle: int,
    samples: tuple[dict[str, object], ...],
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
    *,
    fixture_only: bool,
) -> LedgerDocument:
    """Bind the accepted sample identities before any inference attempt."""
    return {
        "kind": "samples",
        "cycle": cycle,
        "sample_ids": [sample["record_id"] for sample in samples],
        "sample_digests": [sample["digest"] for sample in samples],
        "sample_records": [dict(sample) for sample in samples],
        "policy_digest": policy.canonical_digest,
        "evidence_scope": "test_fixture_only" if fixture_only else "production",
    }


def inference_decision(cycle: int, receipt: InferenceReceipt) -> LedgerDocument:
    """Persist the complete inference receipt and explicit action-zero selection."""
    policy_evidence = (
        _FIXTURE_POLICY_EVIDENCE
        if receipt.policy == "fixture_deterministic_adapter"
        else _FROZEN_POLICY_EVIDENCE
    )
    return {
        "kind": "inference",
        "cycle": cycle,
        "inference_digest": receipt_digest(receipt),
        "inference_receipt": receipt.to_document(),
        "selected_action_0": receipt.action_chunk_float32_2d[0],
        "policy_evidence": policy_evidence,
    }


def cartesian_decision(
    cycle: int,
    receipt: CartesianProposalReceipt,
    camera_corpus: Mapping[str, object],
) -> LedgerDocument:
    """Persist every field of the physical task-frame transform receipt."""
    document = {
        "raw_xy": list(receipt.raw_xy),
        "raw_xyz": list(receipt.raw_xyz),
        "applied_xyz": list(receipt.applied_xyz),
        "tool_rpy": list(receipt.tool_rpy),
        "transform_hash": receipt.transform_hash,
        "camera_digest": receipt.camera_digest,
        "policy_digest": receipt.policy_digest,
        "clipping_performed": receipt.clipping_performed,
        "ik_called": receipt.ik_called,
    }
    material = {
        "physical_to_sim_se2": camera_corpus["physical_to_sim_se2"],
        "camera_digest": receipt.camera_digest,
    }
    return {
        "kind": "cartesian_transform",
        "cycle": cycle,
        "transform_hash": receipt.transform_hash,
        "cartesian_receipt_hash": canonical_hash(document),
        "transform_material": material,
        "cartesian_receipt": document,
    }


def ik_decision(cycle: int, proposal: PhysicalIKProposal) -> LedgerDocument:
    """Persist the complete physical-IK proposal and its canonical hash."""
    return {
        "kind": "ik_proposal",
        "cycle": cycle,
        "proposal_hash": proposal.proposal_hash,
        "ik_proposal": proposal.to_document(),
    }


def supervisor_acceptance(cycle: int, token: AuthorizationToken) -> LedgerDocument:
    """Persist the supervisor decision and complete non-consumed token receipt."""
    return {
        "kind": "supervisor_decision",
        "cycle": cycle,
        "decision": "ACCEPT",
        "authorization_token": {
            "token_id": token.token_id,
            "proposal_hash": token.proposal_hash,
            "policy_digest": token.policy_digest,
            "command_id": token.command_id,
            "valid_until": token.valid_until,
            "digest": token.digest,
        },
    }


def supervisor_rejection(
    cycle: int,
    exc: RolloutViolation,
    *,
    after_stage: str,
) -> LedgerDocument:
    """Persist the exact rejection at the last durable decision stage."""
    return {
        "kind": "supervisor_decision",
        "cycle": cycle,
        "decision": "REJECT",
        "after_stage": after_stage,
        "rejection_code": exc.code.value,
        "rejection_detail": str(exc),
    }
