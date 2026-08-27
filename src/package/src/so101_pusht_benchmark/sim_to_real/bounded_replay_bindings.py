"""Signed-authority, scene, freshness, and terminal bindings for bounded replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
import math
from typing import cast

from .bounded_authorization import BoundedAuthorization, verify_bounded_authorization_document
from .ledger_chain import canonical_hash
from .physical_ik_scene_pose import scene_pose_content_digest
from .rollout_codes import RolloutCode, RolloutViolation


def _fail(detail: str) -> RolloutViolation:
    return RolloutViolation(RolloutCode.R_HASH_MISMATCH, detail)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail(label)
    return cast("Mapping[str, object]", value)


@dataclass(frozen=True, slots=True)
class CycleIdentity:
    proposal_hash: str
    inference_digest: str
    observation_digest: str
    sample_ids: tuple[str, str]
    sample_digests: tuple[str, str]
    token_id: str
    token_digest: str


def verify_signed_authority(record: Mapping[str, object]) -> BoundedAuthorization:
    """Recompute the persisted signature and bind every ledger budget field."""
    signed = _mapping(record.get("signed_authorization"), "signed bounded authorization")
    single_digest = signed.get("single_step_receipt_digest")
    if not isinstance(single_digest, str):
        raise _fail("signed single-step promotion digest")
    authority = verify_bounded_authorization_document(
        signed,
        now=None,
        single_step_receipt_digest=single_digest,
    )
    bindings = (
        (record.get("authorization_digest"), authority.digest),
        (record.get("policy_digest"), authority.policy_digest),
        (record.get("max_commands"), authority.max_commands),
        (record.get("max_duration_seconds"), authority.max_duration_seconds),
        (record.get("max_path_length_m"), authority.max_path_length_m),
        (record.get("max_error_count"), authority.max_error_count),
    )
    if any(actual != expected for actual, expected in bindings):
        raise _fail("signed bounded authorization ledger binding")
    return authority


def verify_cycle_scene(
    samples: Mapping[str, object],
    proposal: Mapping[str, object],
    cartesian: Mapping[str, object],
    inference: Mapping[str, object],
    token: Mapping[str, object],
) -> CycleIdentity:
    """Bind authenticated obstacle pose and proposal to the exact second sample."""
    raw_samples = samples.get("sample_records")
    if not isinstance(raw_samples, list):
        raise _fail("bounded scene sample pair")
    typed_samples = cast("list[object]", raw_samples)
    if len(typed_samples) != 2:
        raise _fail("bounded scene sample pair")
    parsed = [_mapping(value, "bounded scene sample") for value in typed_samples]
    second = parsed[-1]
    pose = _mapping(samples.get("scene_pose_receipt"), "bounded scene pose receipt")
    cartesian_receipt = _mapping(cartesian.get("cartesian_receipt"), "bounded Cartesian receipt")
    digest = pose.get("digest")
    if not isinstance(digest, str) or scene_pose_content_digest(pose) != digest:
        raise _fail("bounded scene pose content")
    bindings = (
        (pose.get("sample_id"), second.get("record_id")),
        (pose.get("sample_timestamp"), second.get("created_at")),
        (pose.get("sample_digest"), second.get("digest")),
        (pose.get("device_digest"), second.get("device_digest")),
        (pose.get("camera_registration_digest"), cartesian_receipt.get("camera_digest")),
        (pose.get("policy_digest"), proposal.get("policy_digest")),
        (pose.get("model_digest"), proposal.get("model_digest")),
        (digest, proposal.get("scene_pose_digest")),
    )
    if any(actual != expected for actual, expected in bindings):
        raise _fail("bounded scene pose cycle binding")
    ids = tuple(str(sample.get("record_id")) for sample in parsed)
    digests = tuple(str(sample.get("digest")) for sample in parsed)
    return CycleIdentity(
        str(proposal.get("proposal_hash")),
        str(inference.get("inference_digest")),
        canonical_hash(
            {
                "observations": [
                    [sample.get("frame_digest"), sample.get("body_degrees")] for sample in parsed
                ]
            }
        ),
        cast("tuple[str, str]", ids),
        cast("tuple[str, str]", digests),
        str(token.get("token_id")),
        str(token.get("digest")),
    )


def verify_fresh_cycles(cycles: Sequence[CycleIdentity]) -> None:
    """Reject replayed proposals, observations, inference authority, or tokens."""
    tokens = [(cycle.token_id, cycle.token_digest) for cycle in cycles]
    sample_members = [
        member for cycle in cycles for member in (*cycle.sample_ids, *cycle.sample_digests)
    ]
    if len(tokens) != len(set(tokens)) or len(sample_members) != len(set(sample_members)):
        raise _fail("bounded sample or authorization identity reused across cycles")
    for cycle in cycles:
        repeated = [item for item in cycles if item.proposal_hash == cycle.proposal_hash]
        if len(repeated) > 1 and (
            len({item.observation_digest for item in repeated}) != 1
            or len({item.inference_digest for item in repeated}) != len(repeated)
        ):
            raise _fail("bounded proposal identity reused across distinct observations")


def _usage(
    records: Sequence[Mapping[str, object]], key: str, *, monotonic: bool = True
) -> list[float]:
    values: list[float] = []
    for record in records:
        if record.get("kind") != "budget_accounting":
            continue
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _fail(f"bounded budget accounting {key}")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0:
            raise _fail(f"bounded budget accounting {key}")
        values.append(parsed)
    if not values or (monotonic and any(right < left for left, right in pairwise(values))):
        raise _fail(f"bounded budget accounting {key} sequence")
    return values


def verify_terminal_authority(
    records: Sequence[Mapping[str, object]],
    terminal: Mapping[str, object],
    fault_records: Sequence[Mapping[str, object]],
    authority: BoundedAuthorization,
) -> None:
    """Recompute all signed budgets and require coherent terminal promotion."""
    commands = max(
        sum(record.get("kind") == "intent" for record in records),
        int(max(_usage(records, "command_count"))),
    )
    errors = max(
        sum(record.get("kind") == "error_event" for record in records),
        int(max(_usage(records, "error_count"))),
    )
    elapsed = max(_usage(records, "elapsed_seconds"))
    paths = _usage(records, "cumulative_path_m")
    increments = sum(_usage(records, "swept_path_increment_m", monotonic=False)) + sum(
        _usage(records, "target_transition_increment_m", monotonic=False)
    )
    if not math.isclose(max(paths), increments, rel_tol=0.0, abs_tol=1e-12):
        raise _fail("bounded cumulative path accounting")
    breaches = (
        commands > authority.max_commands,
        elapsed > authority.max_duration_seconds,
        max(paths) > authority.max_path_length_m,
        errors > authority.max_error_count,
    )
    if (
        terminal.get("authorization_digest") != authority.digest
        or terminal.get("max_error_count") != authority.max_error_count
        or terminal.get("error_count") != errors
    ):
        raise _fail("bounded terminal signed-authority binding")
    cleanup = [record for record in records if record.get("kind") == "cleanup"]
    persisted_budget_fault = any(
        record.get("fault_code") == RolloutCode.R_BUDGET_EXHAUSTED.value
        for record in (*fault_records, terminal)
    )
    if any(breaches) or persisted_budget_fault:
        if len(fault_records) != 1 or len(cleanup) != 1:
            raise _fail("bounded budget fault records")
        detail = fault_records[0].get("fault_detail")
        required = ("FAULT", RolloutCode.R_BUDGET_EXHAUSTED.value, detail)
        for record in (terminal, cleanup[0]):
            if (
                record.get("state"),
                record.get("fault_code"),
                record.get("fault_detail"),
            ) != required:
                raise _fail("bounded budget fault terminal/detail binding")
        if not isinstance(detail, str) or not detail:
            raise _fail("bounded budget fault detail")
        return
    state, fault = terminal.get("state"), terminal.get("fault_code")
    if state == "COMPLETE":
        if fault is not None or fault_records:
            raise _fail("bounded COMPLETE status is incoherent")
    elif state != "FAULT" or fault is None or not fault_records:
        raise _fail("bounded terminal status")
