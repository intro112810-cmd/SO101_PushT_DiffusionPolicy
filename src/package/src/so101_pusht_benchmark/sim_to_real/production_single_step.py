"""Prepared production single-step execution through the sole writer."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from .arming import ArmingResult
from .authorization import AuthorizationToken
from .ledger_chain import canonical_hash
from .physical_ik import PhysicalIKProposal
from .rollout_codes import RolloutCode, RolloutViolation
from .single_step_authorization import SingleStepAuthorization
from .single_step_evidence import (
    AcknowledgementProvider,
    PostStateProvider,
    validate_acknowledgement,
    validate_post_state,
)
from .writer import DirectBusRobot, DirectBusWriter, DispatchIntent


class IntentSink(Protocol):
    def __call__(self, intent: DispatchIntent) -> None: ...


@dataclass(frozen=True, slots=True)
class PreparedProductionSingleStep:
    authorization: SingleStepAuthorization
    armed: ArmingResult
    token: AuthorizationToken
    proposal: PhysicalIKProposal
    pre_sample_digests: frozenset[str]
    newer_than: float = 999.97


def _discard_intent(intent: DispatchIntent) -> None:
    del intent


@dataclass(frozen=True, slots=True)
class ProductionSingleStepRuntime:
    robot: DirectBusRobot
    acknowledgement_provider: AcknowledgementProvider
    post_state_provider: PostStateProvider
    append_intent: IntentSink = _discard_intent


class _PreparedAuthority:
    """Mutable single-use consumer whose only purpose is budget consumption."""

    def __init__(self, prepared: PreparedProductionSingleStep) -> None:
        self._prepared = prepared
        self._consumed = False

    def consume(self, token: AuthorizationToken, proposal_hash: str, command_id: str) -> None:
        prepared = self._prepared
        authorization = prepared.authorization
        armed = prepared.armed
        if self._consumed:
            raise RolloutViolation(RolloutCode.R_DUPLICATE_DISPATCH, "authorization consumed")
        if (
            authorization.artifact_scope != "production"
            or token != prepared.token
            or proposal_hash != authorization.proposal_hash
            or command_id != authorization.command_id
            or token.policy_digest != authorization.policy_digest
            or armed.receipt_digest != authorization.armed_receipt_digest
            or armed.authorization_digest != authorization.digest
        ):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "production execution binding")
        self._consumed = True


def execute_production_single_step(
    prepared: PreparedProductionSingleStep, runtime: ProductionSingleStepRuntime
) -> dict[str, object]:
    """Dispatch once, validate independent evidence, and return a promotable receipt."""
    writer = DirectBusWriter(runtime.robot, _PreparedAuthority(prepared), runtime.append_intent)
    writer.dispatch(prepared.token, prepared.proposal, prepared.authorization.command_id)
    acknowledgement = runtime.acknowledgement_provider.acknowledge()
    validate_acknowledgement(
        acknowledgement,
        command_id=prepared.authorization.command_id,
        proposal_hash=prepared.proposal.proposal_hash,
        body_degrees=prepared.proposal.body_degrees,
        newer_than=prepared.newer_than,
    )
    post_state = runtime.post_state_provider.read_post_state()
    validate_post_state(
        post_state,
        command_id=prepared.authorization.command_id,
        acknowledgement=acknowledgement,
        pre_sample_digests=prepared.pre_sample_digests,
    )
    document: dict[str, object] = {
        "schema": 2,
        "state": "COMPLETE",
        "write_count": 1,
        "command_id": prepared.authorization.command_id,
        "proposal_hash": prepared.proposal.proposal_hash,
        "authorization_digest": prepared.authorization.digest,
        "proposal": prepared.proposal.to_document(),
        "pre_sample_digests": sorted(prepared.pre_sample_digests),
        "acknowledgement": acknowledgement.content() | {"digest": acknowledgement.digest},
        "post_state": post_state.content() | {"digest": post_state.digest},
    }
    document["digest"] = canonical_hash(document)
    return document
