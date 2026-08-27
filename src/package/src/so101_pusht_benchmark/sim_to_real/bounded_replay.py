"""Deterministic semantic replay of bounded per-cycle evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from .authorization import AuthorizationToken, verify_authorization
from .bounded_replay_bindings import (
    CycleIdentity,
    verify_cycle_scene,
    verify_fresh_cycles,
    verify_signed_authority,
    verify_terminal_authority,
)
from .ledger_chain import GENESIS_DIGEST, verify_ledger
from .rollout_codes import RolloutCode, RolloutViolation
from .shadow_ledger import append_record
from .shadow_replay import verify_shadow_decision_ledger
from .single_step_evidence import (
    AcknowledgementEvidence,
    PostStateEvidence,
    validate_acknowledgement,
    validate_post_state,
)

_CORE = {"samples", "inference", "cartesian_transform", "ik_proposal", "supervisor_decision"}


def _fail(detail: str) -> RolloutViolation:
    return RolloutViolation(RolloutCode.R_HASH_MISMATCH, detail)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail(label)
    return cast("Mapping[str, object]", value)


def _items(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise _fail(label)
    return cast("list[object]", value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(label)
    return float(value)


def _cycle(
    records: Sequence[Mapping[str, object]],
    cycle: int,
    authority: Mapping[str, object],
) -> CycleIdentity:
    core = [
        dict(record)
        for record in records
        if record.get("cycle") == cycle and record.get("kind") in _CORE
    ]
    if [record.get("kind") for record in core] != [
        "samples",
        "inference",
        "cartesian_transform",
        "ik_proposal",
        "supervisor_decision",
    ]:
        raise _fail("bounded core stage order")
    synthetic: list[dict[str, object]] = []
    previous = GENESIS_DIGEST
    for record in core:
        record.pop("prev_digest", None)
        record.pop("digest", None)
        record.pop("sequence", None)
        previous = append_record(synthetic, record, previous_digest=previous)
    append_record(
        synthetic,
        {
            "kind": "cleanup",
            "status": "released",
            "terminal_state": "SHADOW_COMPLETE",
            "writer_closed": True,
            "motor_writes_performed": False,
            "actuation_performed": False,
            "writer_symbols": 0,
            "read_only": True,
        },
        previous_digest=previous,
    )
    verify_shadow_decision_ledger(synthetic)
    samples = core[0]
    inference = core[1]
    cartesian = core[2]
    proposal = cast("Mapping[str, object]", core[3]["ik_proposal"])
    token_document = _mapping(core[4]["authorization_token"], "bounded token")
    identity = verify_cycle_scene(samples, proposal, cartesian, inference, token_document)
    token = AuthorizationToken(
        str(token_document.get("token_id")),
        str(token_document.get("proposal_hash")),
        str(token_document.get("policy_digest")),
        str(token_document.get("command_id")),
        _number(token_document.get("valid_until"), "bounded token expiry"),
        str(token_document.get("digest")),
    )
    verify_authorization(token)
    intent = next(
        (
            record
            for record in records
            if record.get("cycle") == cycle and record.get("kind") == "intent"
        ),
        None,
    )
    ack_raw = next(
        (
            record
            for record in records
            if record.get("cycle") == cycle and record.get("kind") == "acknowledgement"
        ),
        None,
    )
    post_raw = next(
        (
            record
            for record in records
            if record.get("cycle") == cycle and record.get("kind") == "post_state"
        ),
        None,
    )
    fault_record = next(
        (
            record
            for record in records
            if record.get("cycle") == cycle and record.get("kind") == "cycle_fault"
        ),
        None,
    )
    discarded = next(
        (
            record
            for record in records
            if record.get("cycle") == cycle and record.get("kind") == "discarded_actions"
        ),
        None,
    )
    if intent is None or ack_raw is None or post_raw is None:
        raise _fail("bounded execution stage missing")
    if discarded is None and fault_record is None:
        raise _fail("bounded terminal cycle decision missing")
    exact_bindings = (
        (intent.get("command_id"), token.command_id),
        (intent.get("proposal_hash"), token.proposal_hash),
        (proposal.get("proposal_hash"), token.proposal_hash),
        (intent.get("token_digest"), token.digest),
        (intent.get("token_id"), token.token_id),
        (intent.get("policy_digest"), token.policy_digest),
        (intent.get("valid_until"), token.valid_until),
        (intent.get("command_budget"), 1),
        (intent.get("bounded_authorization_digest"), authority.get("authorization_digest")),
        (token.policy_digest, authority.get("policy_digest")),
    )
    if any(actual != expected for actual, expected in exact_bindings):
        raise _fail("bounded intent/token/authorization binding")
    ack_evidence = ack_raw.get("evidence")
    post_evidence = post_raw.get("evidence")
    if not isinstance(ack_evidence, dict) or not isinstance(post_evidence, dict):
        raise _fail("bounded provider evidence")
    ack = AcknowledgementEvidence.from_mapping(cast("dict[str, object]", ack_evidence))
    body_items = _items(proposal.get("body_degrees"), "bounded proposal body")
    if len(body_items) != 5:
        raise _fail("bounded proposal body")
    body = cast(
        "tuple[float, float, float, float, float]",
        tuple(_number(value, "bounded proposal body") for value in body_items),
    )
    sample_records = _items(samples.get("sample_records"), "bounded samples")
    if len(sample_records) != 2:
        raise _fail("bounded samples")
    parsed_samples = [_mapping(sample, "bounded sample") for sample in sample_records]
    newer_than = max(
        _number(sample.get("created_at"), "bounded sample time") for sample in parsed_samples
    )
    if token.valid_until <= newer_than:
        raise _fail("bounded token expiry binding")
    post = PostStateEvidence.from_mapping(cast("dict[str, object]", post_evidence))
    try:
        validate_acknowledgement(
            ack,
            command_id=str(intent["command_id"]),
            proposal_hash=str(proposal["proposal_hash"]),
            body_degrees=body,
            newer_than=newer_than,
        )
        validate_post_state(
            post,
            command_id=str(intent["command_id"]),
            acknowledgement=ack,
            pre_sample_digests=frozenset(str(sample["digest"]) for sample in parsed_samples),
        )
    except RolloutViolation as exc:
        if fault_record is not None and fault_record.get("fault_code") == exc.code.value:
            return identity
        raise
    if fault_record is not None:
        if (
            fault_record.get("fault_code") == RolloutCode.R_POST_STATE_MISMATCH.value
            and post.body_degrees != body
        ):
            return identity
        raise _fail("bounded fault does not replay")
    if discarded is None:
        raise _fail("bounded discarded actions missing")
    if discarded.get("discarded_indices") != list(range(1, 8)) or discarded.get(
        "inference_digest"
    ) != inference.get("inference_digest"):
        raise _fail("bounded discarded-action binding")
    return identity


def verify_bounded_ledger(records: Sequence[Mapping[str, object]]) -> str:
    """Verify chain, complete stage bindings, and each typed provider receipt."""
    terminal = verify_ledger(records)
    authorities = [record for record in records if record.get("kind") == "bounded_authorization"]
    if len(authorities) != 1:
        raise _fail("bounded authorization record")
    authority_record = authorities[0]
    signed_authority = verify_signed_authority(authority_record)
    cycles = sorted(
        {cast(int, record["cycle"]) for record in records if record.get("kind") == "samples"}
    )
    identities = [_cycle(records, cycle, authority_record) for cycle in cycles]
    verify_fresh_cycles(identities)
    terminal_records = [record for record in records if record.get("kind") == "terminal"]
    fault_records = [record for record in records if record.get("kind") == "cycle_fault"]
    if len(terminal_records) != 1 or terminal_records[0].get("write_count") != len(cycles):
        raise _fail("bounded terminal count")
    terminal_record = terminal_records[0]
    error_events = [record for record in records if record.get("kind") == "error_event"]
    counts = [record.get("error_count") for record in error_events]
    if counts != list(range(1, len(error_events) + 1)):
        raise _fail("bounded error count sequence")
    verify_terminal_authority(
        records,
        terminal_record,
        fault_records,
        signed_authority,
    )
    return terminal
