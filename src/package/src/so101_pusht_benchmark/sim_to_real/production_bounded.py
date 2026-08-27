"""Provider-driven production bounded rollout with one-action cycles."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Protocol
from .authorization import AuthorizationToken
from .physical_ik import PhysicalIKProposal
from .production_single_step import ProductionSingleStepRuntime
from .rollout_codes import RolloutCode, RolloutViolation
from .single_step_evidence import validate_acknowledgement, validate_post_state
from .writer import DirectBusWriter

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class ProductionCycle:
    token: AuthorizationToken
    proposal: PhysicalIKProposal
    pre_sample_digests: frozenset[str]
    newer_than: float
    runtime: ProductionSingleStepRuntime


@dataclass(frozen=True, slots=True)
class ProductionBoundedBudget:
    max_commands: int
    max_duration_seconds: float
    max_path_length_m: float
    max_error_count: int


@dataclass(frozen=True, slots=True)
class ProductionBoundedResult:
    state: str
    write_count: int
    command_ids: tuple[str, ...]
    fault_code: str | None
    error_count: int


class ProductionCycleProvider(Protocol):
    def next_cycle(self, index: int, previous_evidence_digest: str) -> ProductionCycle | None: ...


class _CycleAuthority:
    """Single-use consumer for one supervisor-minted bounded cycle token."""

    def __init__(self, token: AuthorizationToken) -> None:
        self._token = token
        self._used = False

    def consume(self, token: AuthorizationToken, proposal_hash: str, command_id: str) -> None:
        if self._used:
            raise RolloutViolation(RolloutCode.R_DUPLICATE_DISPATCH, "cycle token consumed")
        if (
            token != self._token
            or proposal_hash != token.proposal_hash
            or command_id != token.command_id
        ):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "bounded cycle token")
        self._used = True


def _path_length(proposal: PhysicalIKProposal) -> float:
    return sum(
        math.dist(left, right)
        for left, right in zip(proposal.swept_path, proposal.swept_path[1:], strict=False)
    )


def _fault(commands: list[str], code: RolloutCode, errors: int) -> ProductionBoundedResult:
    return ProductionBoundedResult("FAULT", len(commands), tuple(commands), code.value, errors)


def execute_production_bounded(
    provider: ProductionCycleProvider,
    budget: ProductionBoundedBudget,
    *,
    clock: Clock,
    initial_evidence_digest: str = "0" * 64,
) -> ProductionBoundedResult:
    """Request fresh digest-chained cycles and stop on the first uncertain outcome."""
    started = float(clock())
    commands: list[str] = []
    seen_samples: set[str] = set()
    seen_proposals: set[str] = set()
    path_length = 0.0
    errors = 0
    previous_evidence = initial_evidence_digest
    for index in range(budget.max_commands):
        if float(clock()) - started > budget.max_duration_seconds:
            return _fault(commands, RolloutCode.R_BUDGET_EXHAUSTED, errors)
        cycle = provider.next_cycle(index, previous_evidence)
        if cycle is None:
            return _fault(commands, RolloutCode.R_MISSING, errors)
        if seen_samples & cycle.pre_sample_digests:
            return _fault(commands, RolloutCode.R_DUPLICATE_SAMPLE, errors)
        if cycle.proposal.proposal_hash in seen_proposals:
            return _fault(commands, RolloutCode.R_HASH_MISMATCH, errors)
        path_length += _path_length(cycle.proposal)
        if path_length > budget.max_path_length_m:
            return _fault(commands, RolloutCode.R_BUDGET_EXHAUSTED, errors)
        try:
            DirectBusWriter(
                cycle.runtime.robot,
                _CycleAuthority(cycle.token),
                cycle.runtime.append_intent,
            ).dispatch(cycle.token, cycle.proposal, cycle.token.command_id)
            acknowledgement = cycle.runtime.acknowledgement_provider.acknowledge()
            validate_acknowledgement(
                acknowledgement,
                command_id=cycle.token.command_id,
                proposal_hash=cycle.proposal.proposal_hash,
                body_degrees=cycle.proposal.body_degrees,
                newer_than=cycle.newer_than,
            )
            post_state = cycle.runtime.post_state_provider.read_post_state()
            validate_post_state(
                post_state,
                command_id=cycle.token.command_id,
                acknowledgement=acknowledgement,
                pre_sample_digests=cycle.pre_sample_digests,
            )
        except (RolloutViolation, RuntimeError) as exc:
            errors += 1
            code = exc.code if isinstance(exc, RolloutViolation) else RolloutCode.F_PROVIDER_ERROR
            return _fault(commands, code, errors)
        seen_samples.update(cycle.pre_sample_digests)
        seen_proposals.add(cycle.proposal.proposal_hash)
        previous_evidence = post_state.digest
        commands.append(cycle.token.command_id)
    return ProductionBoundedResult("COMPLETE", len(commands), tuple(commands), None, errors)
