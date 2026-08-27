"""Durable intent and fake-provider evidence for bounded direct-bus cycles."""

from __future__ import annotations

from itertools import pairwise
import json
import math
from pathlib import Path
from typing import cast

from .bounded_pipeline import PlannedCycle
from .ledger_chain import canonical_hash
from .receipt_routing import locate_receipt_path
from .rollout_codes import RolloutCode, RolloutViolation
from .secure_io import ExclusiveAppendFile
from .single_step import SingleStepLedger
from .single_step_evidence import AcknowledgementEvidence, PostStateEvidence
from .writer import DispatchIntent


class DurableIntentLedger(SingleStepLedger):
    """Fsync each writer intent before its sole bus invocation."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._location = locate_receipt_path(path)
        self._stream = ExclusiveAppendFile(self._location.resolved)

    def append(self, intent: DispatchIntent) -> None:
        super().append(intent)
        document = {
            "command_id": intent.command_id,
            "proposal_hash": intent.proposal_hash,
            "body_degrees": list(intent.body_degrees),
        }
        self._stream.append((json.dumps(document, sort_keys=True) + "\n").encode())
        if locate_receipt_path(self._location.lexical) != self._location:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "intent ledger parent changed")


def provider_evidence(
    plan: PlannedCycle,
    command_id: str,
    observed_at: float,
    *,
    provider_modified: bool,
    tracking_fault: bool,
) -> tuple[AcknowledgementEvidence, PostStateEvidence]:
    """Mint independent typed fake-provider evidence for direct tests only."""
    body = list(plan.proposal.body_degrees)
    if provider_modified:
        body[0] += 0.5
    ack_content = {
        "accepted_body_degrees": body,
        "command_id": command_id,
        "observed_at": observed_at,
        "proposal_hash": plan.proposal.proposal_hash,
        "provider_digest": canonical_hash({"provider": command_id}),
    }
    ack = AcknowledgementEvidence.from_mapping(
        {**ack_content, "digest": canonical_hash(ack_content)}
    )
    post_body = list(body)
    if tracking_fault:
        post_body[0] += 2.0
    post_content = {
        "acknowledgement_digest": ack.digest,
        "body_degrees": post_body,
        "command_id": command_id,
        "created_at": observed_at + 0.001,
        "frame_digest": canonical_hash({"post-frame": command_id}),
        "sample_digest": canonical_hash({"post-sample": command_id}),
    }
    post = PostStateEvidence.from_mapping({**post_content, "digest": canonical_hash(post_content)})
    return ack, post


def cartesian_target(plan: PlannedCycle) -> tuple[float, float, float]:
    """Read the applied Cartesian target from the corrected transform receipt."""
    raw_record = plan.records[2].get("cartesian_receipt")
    if not isinstance(raw_record, dict):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "Cartesian receipt")
    record = cast("dict[str, object]", raw_record)
    raw_applied = record.get("applied_xyz")
    if not isinstance(raw_applied, list):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "Cartesian target")
    applied = cast("list[object]", raw_applied)
    if len(applied) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in applied
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "Cartesian target")
    numeric = cast("list[int | float]", applied)
    return float(numeric[0]), float(numeric[1]), float(numeric[2])


def swept_path_length(plan: PlannedCycle) -> float:
    """Measure the real physical-IK swept path without estimating or clipping."""
    points = plan.proposal.swept_path
    return sum(math.dist(left, right) for left, right in pairwise(points))
