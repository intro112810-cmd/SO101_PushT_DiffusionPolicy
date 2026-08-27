"""Extract exact planner decisions into guarded execution packages."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import cast
from .rollout_codes import RolloutCode, RolloutViolation


def _record(records: Sequence[Mapping[str, object]], kind: str, cycle: int) -> Mapping[str, object]:
    matches = [item for item in records if item.get("kind") == kind and item.get("cycle") == cycle]
    if len(matches) != 1:
        raise RolloutViolation(RolloutCode.R_MISSING, f"exact {kind} record required")
    return matches[0]


def build_execution_package(
    records: Sequence[Mapping[str, object]], *, cycle: int, previous_evidence_digest: str | None
) -> dict[str, object]:
    """Bind same-cycle samples, proposal, and supervisor token without mutation."""
    samples = _record(records, "samples", cycle)
    proposal_record = _record(records, "ik_proposal", cycle)
    supervisor = _record(records, "supervisor_decision", cycle)
    if supervisor.get("decision") != "ACCEPT":
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "supervisor did not accept")
    raw_samples = samples.get("sample_records")
    proposal = proposal_record.get("ik_proposal")
    token = supervisor.get("authorization_token")
    if (
        not isinstance(raw_samples, list)
        or not isinstance(proposal, Mapping)
        or not isinstance(token, Mapping)
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "planner decision records")
    sample_records = cast("list[object]", raw_samples)
    if len(sample_records) != 2:
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "planner sample count")
    proposal_mapping = cast("Mapping[str, object]", proposal)
    token_mapping = cast("Mapping[str, object]", token)
    digests: list[str] = []
    times: list[float] = []
    for raw in sample_records:
        if not isinstance(raw, Mapping):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "sample record")
        sample = cast("Mapping[str, object]", raw)
        digest = sample.get("digest")
        created = sample.get("created_at")
        if (
            not isinstance(digest, str)
            or isinstance(created, bool)
            or not isinstance(created, (int, float))
        ):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "sample identity")
        digests.append(digest)
        times.append(float(created))
    package: dict[str, object] = {
        "schema": "production-single-step-package-v1",
        "proposal": dict(proposal_mapping),
        "token": dict(token_mapping),
        "pre_sample_digests": digests,
        "newer_than": max(times),
    }
    if previous_evidence_digest is not None:
        package.update(
            schema="production-bounded-cycle-v1",
            cycle=cycle,
            previous_evidence_digest=previous_evidence_digest,
        )
    return package
