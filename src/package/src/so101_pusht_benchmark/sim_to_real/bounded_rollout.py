"""Bounded one-action receding-horizon orchestration over corrected seams."""

from __future__ import annotations

from . import bounded_authorization, bounded_budget
from .bounded_execution import DurableIntentLedger, provider_evidence
from .bounded_finalize import BoundedFinalization, finalize_bounded_rollout
from .bounded_input import check_bounded_budgets, load_bounded_cycles
from .bounded_pipeline import CyclePlanningInput, plan_fixture_cycle
from .bounded_types import BoundedRolloutInput, BoundedRolloutResult
from .policy_parser import load_fixture_safety_policy
from .receipt_routing import prepare_receipt_directory, validate_receipt_path
from .rollout_codes import RolloutCode, RolloutViolation
from .shadow_ledger import append_record
from .single_step import FixtureBus, count_writes, dispatch_once
from .single_step_evidence import validate_acknowledgement, validate_post_state

__all__ = ("BoundedRolloutInput", "BoundedRolloutResult", "run_fixture_bounded_rollout")


def run_fixture_bounded_rollout(inputs: BoundedRolloutInput) -> BoundedRolloutResult:
    """Re-observe and execute exactly action zero until completion or first fault."""
    validate_receipt_path(inputs.output_dir / "receipt.json", production=False)
    single_digest = bounded_authorization.verify_single_step_receipt(
        inputs.single_step_receipt_path
    )
    authorization = bounded_authorization.load_bounded_authorization(
        inputs.authorization_path, now=inputs.now, single_step_receipt_digest=single_digest
    )
    policy = load_fixture_safety_policy(inputs.policy_path, now=inputs.now)
    if authorization.policy_digest != policy.canonical_digest:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorization policy binding")
    check_bounded_budgets(authorization, policy)
    mode, cycles = load_bounded_cycles(inputs.fixture_dir)
    bus = inputs.robot or FixtureBus()
    if count_writes(bus):
        raise RolloutViolation(RolloutCode.R_DUPLICATE_DISPATCH, "bounded authority reused")
    output_dir = prepare_receipt_directory(inputs.output_dir, production=False)
    intent_ledger = DurableIntentLedger(output_dir / "intents.jsonl")
    started = float(inputs.clock())
    records: list[dict[str, object]] = []
    previous = bounded_budget.record_bounded_authority(records, authorization)
    command_ids: list[str] = []
    seen_samples: set[str] = set()
    seen_proposals: set[str] = set()
    seen_actions: set[str] = set()
    budget = bounded_budget.BudgetRecorder(records)
    fault: str | None = None
    fault_detail: str | None = None
    error_count = 0
    for index, cycle in enumerate(cycles):
        if len(command_ids) >= authorization.max_commands:
            break
        sample_pair = cycle.samples
        elapsed = float(inputs.clock()) - started
        previous = budget.pre_cycle(previous, index, len(command_ids), elapsed, error_count)
        if elapsed > authorization.max_duration_seconds:
            fault = RolloutCode.R_BUDGET_EXHAUSTED.value
            fault_detail = "elapsed time exceeds signed budget"
            break
        identities = {str(item["record_id"]) for item in sample_pair} | {
            str(item["digest"]) for item in sample_pair
        }
        if seen_samples & identities:
            fault = RolloutCode.R_DUPLICATE_SAMPLE.value
            break
        if mode == "bounded_fault_after_two" and index == 2:
            fault = RolloutCode.F_PROVIDER_ERROR.value
            break
        if (mode == "bounded_one_error" and index == 1) or (
            mode == "bounded_error_breach" and index in {1, 2}
        ):
            error_count += 1
            previous = append_record(
                records,
                {
                    "kind": "error_event",
                    "cycle": index,
                    "error_code": RolloutCode.F_PROVIDER_ERROR.value,
                    "error_count": error_count,
                    "write_count": count_writes(bus),
                },
                previous_digest=previous,
            )
            if error_count > authorization.max_error_count:
                fault = RolloutCode.R_BUDGET_EXHAUSTED.value
                fault_detail = "error count exceeds signed budget"
                break
            continue
        command_id = f"command-{len(command_ids) + 1}"
        try:
            plan = plan_fixture_cycle(
                CyclePlanningInput(
                    index,
                    sample_pair,
                    policy,
                    inputs.clock,
                    command_id,
                    479235,
                    cycle.scene_pose,
                )
            )
            if plan.action_digest in seen_actions:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "static policy output")
            if plan.proposal.proposal_hash in seen_proposals:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "static proposal")
            previous, target = budget.planned_path(previous, plan)
            if budget.path_length_m > authorization.max_path_length_m:
                raise RolloutViolation(RolloutCode.R_BUDGET_EXHAUSTED, "path budget")
            for record in plan.records:
                previous = append_record(records, record, previous_digest=previous)
            previous = append_record(
                records,
                {
                    "kind": "intent",
                    "cycle": index,
                    "command_id": command_id,
                    "proposal_hash": plan.proposal.proposal_hash,
                    "token_digest": plan.token.digest,
                    "token_id": plan.token.token_id,
                    "policy_digest": plan.token.policy_digest,
                    "valid_until": plan.token.valid_until,
                    "command_budget": 1,
                    "bounded_authorization_digest": authorization.digest,
                },
                previous_digest=previous,
            )
            dispatch_once(bus, intent_ledger, plan.token, plan.proposal, command_id)
            ack, post = provider_evidence(
                plan,
                command_id,
                max(item.created_at for item in plan.samples) + 0.001,
                provider_modified=mode == "bounded_provider_modified",
                tracking_fault=mode == "bounded_tracking_fault",
            )
            for kind, document in (
                ("acknowledgement", ack.content() | {"digest": ack.digest}),
                ("post_state", post.content() | {"digest": post.digest}),
            ):
                previous = append_record(
                    records,
                    {"kind": kind, "cycle": index, "evidence": document},
                    previous_digest=previous,
                )
            validate_acknowledgement(
                ack,
                command_id=command_id,
                proposal_hash=plan.proposal.proposal_hash,
                body_degrees=plan.proposal.body_degrees,
                newer_than=max(item.created_at for item in plan.samples),
            )
            validate_post_state(
                post,
                command_id=command_id,
                acknowledgement=ack,
                pre_sample_digests=frozenset(item.digest for item in plan.samples),
            )
            if (
                max(
                    abs(a - b)
                    for a, b in zip(post.body_degrees, plan.proposal.body_degrees, strict=True)
                )
                > policy.post_state.max_tracking_error_degrees
            ):
                raise RolloutViolation(RolloutCode.R_POST_STATE_MISMATCH, "tracking fault")
            previous = append_record(
                records,
                {
                    "kind": "cycle_verified",
                    "cycle": index,
                    "command_id": command_id,
                    "proposal_hash": plan.proposal.proposal_hash,
                    "post_state_digest": post.digest,
                },
                previous_digest=previous,
            )
            previous = append_record(
                records,
                {
                    "kind": "discarded_actions",
                    "cycle": index,
                    "discarded_indices": list(range(1, 8)),
                    "inference_digest": plan.inference_digest,
                },
                previous_digest=previous,
            )
        except (RolloutViolation, RuntimeError) as exc:
            fault = (
                exc.code.value
                if isinstance(exc, RolloutViolation)
                else RolloutCode.F_PROVIDER_ERROR.value
            )
            fault_detail = str(exc)
            error_count += 1
            previous = append_record(
                records,
                {
                    "kind": "error_event",
                    "cycle": index,
                    "error_code": fault,
                    "error_count": error_count,
                    "write_count": count_writes(bus),
                },
                previous_digest=previous,
            )
            previous = append_record(
                records,
                {
                    "kind": "cycle_fault",
                    "cycle": index,
                    "fault_code": fault,
                    "fault_detail": fault_detail,
                    "write_count": count_writes(bus),
                    "retry_count": 0,
                    "compensation_count": 0,
                },
                previous_digest=previous,
            )
            break
        seen_samples.update(identities)
        seen_proposals.add(plan.proposal.proposal_hash)
        seen_actions.add(plan.action_digest)
        budget.accept_target(target)
        command_ids.append(command_id)
    return finalize_bounded_rollout(
        BoundedFinalization(
            records,
            previous,
            fault,
            fault_detail,
            error_count,
            authorization,
            bus,
            command_ids,
            output_dir,
        )
    )
