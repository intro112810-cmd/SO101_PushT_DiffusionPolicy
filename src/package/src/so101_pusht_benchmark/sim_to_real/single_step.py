"""One signed and armed single-step execution through the sole direct-bus writer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Final

from .arming import ArmingCheckInput, ArmingResult, check_arming
from .authorization import AuthorizationToken
from .joint_mapping import JOINT_ORDER
from .physical_ik import PhysicalIKProposal
from .receipt_routing import prepare_receipt_directory, validate_receipt_path
from .rollout_codes import RolloutCode, RolloutViolation
from .single_step_authorization import SingleStepAuthorization, load_single_step_authorization
from .single_step_evidence import (
    AcknowledgementEvidence,
    AcknowledgementProvider,
    PostStateEvidence,
    PostStateProvider,
    load_fixture_evidence_providers,
    validate_acknowledgement,
    validate_post_state,
)
from .single_step_fixture import FixtureBus, fixture_evidence, physical_proposal
from .supervisor import RolloutSupervisor
from .writer import DirectBusWriter, DispatchIntent

__all__ = (
    "AcknowledgementEvidence",
    "FixtureBus",
    "PostStateEvidence",
    "SingleStepBudget",
    "SingleStepRunInput",
    "SingleStepRuntime",
    "check_ack",
    "dispatch_once",
    "fixture_evidence",
    "run_fixture_single_step",
)

_FIXTURE_NOW: Final = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
_ROOT: Final = Path(__file__).resolve().parents[3]
_POLICY: Final = _ROOT / "tests/fixtures/sim_to_real/collision_approved_policy.yaml"
_PROFILE: Final = _ROOT / "configs/hardware/so101_real_v1.yaml"
_SHADOW: Final = _ROOT / "tests/fixtures/sim_to_real/shadow_campaign.jsonl"


class _FixtureRobot:
    def __init__(self, bus: FixtureBus) -> None:
        self.bus = bus


class _FixtureClock:
    def __call__(self) -> float:
        return 1000.0


class _SingleStepLedger:
    def __init__(self) -> None:
        self.intents: list[DispatchIntent] = []

    def append(self, intent: DispatchIntent) -> None:
        self.intents.append(intent)


SingleStepLedger = _SingleStepLedger


@dataclass(frozen=True, slots=True)
class SingleStepRunInput:
    """All paths required for one fixture execution."""

    fixture_path: Path
    authorization_path: Path
    output_dir: Path


@dataclass(slots=True)
class SingleStepBudget:
    """Process-local one-call budget for one signed authorization digest."""

    consumed_digest: str | None = None

    def consume(self, authorization_digest: str) -> None:
        if self.consumed_digest is not None:
            raise RolloutViolation(RolloutCode.R_DUPLICATE_DISPATCH, "authorization consumed")
        self.consumed_digest = authorization_digest


@dataclass(slots=True)
class SingleStepRuntime:
    """Injected fake-bus providers; none can be inferred from the write payload."""

    bus: FixtureBus
    acknowledgement_provider: AcknowledgementProvider
    post_state_provider: PostStateProvider
    budget: SingleStepBudget


class _ExecutionAuthority:
    def __init__(
        self,
        supervisor: RolloutSupervisor,
        authorization: SingleStepAuthorization,
        armed: ArmingResult,
        budget: SingleStepBudget,
    ) -> None:
        self._supervisor = supervisor
        self._authorization = authorization
        self._armed = armed
        self._budget = budget

    def consume(self, token: AuthorizationToken, proposal_hash: str, command_id: str) -> None:
        authorization = self._authorization
        if (
            proposal_hash != authorization.proposal_hash
            or command_id != authorization.command_id
            or self._armed.receipt_digest != authorization.armed_receipt_digest
            or token.policy_digest != authorization.policy_digest
        ):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "authorization binding")
        self._supervisor.consume(token, proposal_hash, command_id)
        self._budget.consume(authorization.digest)


def count_writes(bus: FixtureBus) -> int:
    return sum(entry[0] == "sync_write" for entry in bus.log)


def check_ack(bus: FixtureBus, body_degrees: tuple[float, ...]) -> None:
    """Retain bounded fixture compatibility; single-step execution does not call this."""
    acknowledgement = bus.ack_payload()
    if set(acknowledgement) != set(JOINT_ORDER):
        raise RolloutViolation(RolloutCode.R_ACK_MISMATCH, "ack joint order mismatch")
    for register, degree in zip(JOINT_ORDER, body_degrees, strict=True):
        if acknowledgement.get(register) != degree:
            raise RolloutViolation(RolloutCode.R_ACK_MISMATCH, "ack body degree mismatch")


def _attempt_dispatch(
    writer: DirectBusWriter,
    bus: FixtureBus,
    token: AuthorizationToken,
    proposal: PhysicalIKProposal,
    command_id: str,
) -> None:
    try:
        writer.dispatch(token, proposal, command_id)
    except RuntimeError:
        if count_writes(bus) == 1:
            raise RolloutViolation(
                RolloutCode.R_AMBIGUOUS_DISPATCH,
                "transport outcome unknown after the single write",
            ) from None
        raise


def dispatch_once(
    bus: FixtureBus,
    ledger: SingleStepLedger,
    token: AuthorizationToken,
    proposal: PhysicalIKProposal,
    command_id: str,
) -> None:
    """Compatibility dispatch for bounded fixtures through the sole writer."""
    consumer = _LegacyTokenConsumer(token, proposal.proposal_hash, command_id)
    writer = DirectBusWriter(_FixtureRobot(bus), consumer, ledger.append)
    _attempt_dispatch(writer, bus, token, proposal, command_id)


def run_fixture_single_step(
    inputs: SingleStepRunInput,
    *,
    runtime: SingleStepRuntime | None = None,
) -> dict[str, object]:
    """Consume the shared arming contract and verify one fake-bus fixture command."""
    validate_receipt_path(inputs.output_dir / "receipt.json", production=False)
    fixed_ack, fixed_post, fixture_proposal = load_fixture_evidence_providers(inputs.fixture_path)
    selected = runtime or SingleStepRuntime(FixtureBus(), fixed_ack, fixed_post, SingleStepBudget())
    authorization = load_single_step_authorization(inputs.authorization_path, now=_FIXTURE_NOW)
    armed = check_arming(
        ArmingCheckInput(
            _PROFILE,
            _POLICY,
            _SHADOW,
            inputs.authorization_path,
            inputs.fixture_path / "operational",
            _FIXTURE_NOW,
        )
    )
    proposal = physical_proposal()
    if proposal != fixture_proposal or proposal.proposal_hash != armed.proposal_hash:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "fixture proposal binding")
    body = proposal.body_degrees
    evidence = replace(fixture_evidence(), command_id=armed.command_id)
    supervisor = RolloutSupervisor(_FixtureClock())
    token = supervisor.mint(evidence)
    authority = _ExecutionAuthority(supervisor, authorization, armed, selected.budget)
    writer = DirectBusWriter(_FixtureRobot(selected.bus), authority, _SingleStepLedger().append)
    _attempt_dispatch(writer, selected.bus, token, evidence.ik_proposal, armed.command_id)

    acknowledgement = selected.acknowledgement_provider.acknowledge()
    validate_acknowledgement(
        acknowledgement,
        command_id=armed.command_id,
        proposal_hash=armed.proposal_hash,
        body_degrees=body,
        newer_than=max(sample.created_at for sample in evidence.samples),
    )
    post_state = selected.post_state_provider.read_post_state()
    validate_post_state(
        post_state,
        command_id=armed.command_id,
        acknowledgement=acknowledgement,
        pre_sample_digests=frozenset(sample.digest for sample in evidence.samples),
    )
    result: dict[str, object] = {
        "acknowledgement_digest": acknowledgement.digest,
        "command_id": armed.command_id,
        "evidence_scope": "test_fixture_only",
        "motor_writes_performed": True,
        "policy_evidence": "fixture_fake_bus_not_production",
        "post_state_digest": post_state.digest,
        "state": "COMPLETE",
        "write_count": count_writes(selected.bus),
    }
    output_dir = prepare_receipt_directory(inputs.output_dir, production=False)
    (output_dir / "receipt.json").write_text(
        json.dumps(result, sort_keys=True, indent=2), encoding="utf-8"
    )
    return result


class _LegacyTokenConsumer:
    """Compatibility consumer retained only for bounded fixture orchestration."""

    def __init__(self, token: AuthorizationToken, proposal_hash: str, command_id: str) -> None:
        self._token = token
        self._proposal_hash = proposal_hash
        self._command_id = command_id
        self._consumed = False

    def consume(self, token: AuthorizationToken, proposal_hash: str, command_id: str) -> None:
        if (
            self._consumed
            or token != self._token
            or proposal_hash != self._proposal_hash
            or command_id != self._command_id
        ):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "authorization binding")
        self._consumed = True
