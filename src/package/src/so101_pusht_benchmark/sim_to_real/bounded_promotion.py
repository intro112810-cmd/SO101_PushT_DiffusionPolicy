"""Typed verification of the single-step receipt promoted into bounded authority."""

from __future__ import annotations

from typing import cast

from .ledger_chain import canonical_hash
from .physical_ik_proposal import physical_ik_proposal_hash
from .physical_ik_replay import parse_physical_ik_proposal
from .rollout_codes import RolloutCode, RolloutViolation
from .single_step_evidence import (
    AcknowledgementEvidence,
    PostStateEvidence,
    validate_acknowledgement,
    validate_post_state,
)


def verify_single_step_receipt_document(doc: dict[str, object]) -> str:
    """Revalidate typed acknowledgement, proposal proof, and newer post-state."""
    expected = {
        "schema",
        "state",
        "write_count",
        "command_id",
        "proposal_hash",
        "authorization_digest",
        "proposal",
        "pre_sample_digests",
        "acknowledgement",
        "post_state",
        "digest",
    }
    if set(doc) != expected or doc.get("schema") != 2 or doc.get("state") != "COMPLETE":
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "single-step receipt unverified")
    if doc.get("write_count") != 1:
        raise RolloutViolation(RolloutCode.R_BUDGET_EXHAUSTED, "single-step write count")
    declared = doc.get("digest")
    if (
        not isinstance(declared, str)
        or canonical_hash({key: value for key, value in doc.items() if key != "digest"}) != declared
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "single-step receipt digest")
    proposal_raw, pre_raw = doc.get("proposal"), doc.get("pre_sample_digests")
    if not isinstance(proposal_raw, dict) or not isinstance(pre_raw, list):
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "single-step receipt proposal/history"
        )
    proposal_document = cast("dict[str, object]", proposal_raw)
    proposal_hash = proposal_document.get("proposal_hash")
    if not isinstance(proposal_hash, str):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "single-step proposal hash")
    unhashed = {key: value for key, value in proposal_document.items() if key != "proposal_hash"}
    if physical_ik_proposal_hash(unhashed) != proposal_hash:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "single-step proposal content")
    proposal = parse_physical_ik_proposal(unhashed, declared_hash=proposal_hash)
    if proposal.proposal_hash != doc.get("proposal_hash"):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "single-step proposal binding")
    ack_raw, post_raw = doc.get("acknowledgement"), doc.get("post_state")
    if not isinstance(ack_raw, dict) or not isinstance(post_raw, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, "single-step provider evidence")
    ack = AcknowledgementEvidence.from_mapping(cast("dict[str, object]", ack_raw))
    post = PostStateEvidence.from_mapping(cast("dict[str, object]", post_raw))
    validate_acknowledgement(
        ack,
        command_id=str(doc["command_id"]),
        proposal_hash=proposal_hash,
        body_degrees=proposal.body_degrees,
        newer_than=999.97,
    )
    validate_post_state(
        post,
        command_id=str(doc["command_id"]),
        acknowledgement=ack,
        pre_sample_digests=frozenset(str(value) for value in cast("list[object]", pre_raw)),
    )
    return declared
