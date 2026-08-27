"""Typed provider acknowledgement and readback evidence for one physical command."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Protocol, cast

from .ledger_chain import canonical_hash
from .physical_ik_proposal import PhysicalIKProposal, physical_ik_proposal_hash
from .physical_ik_replay import parse_physical_ik_proposal
from .rollout_codes import RolloutCode, RolloutViolation
from .rollout_record_types import BodyDegrees

__all__ = (
    "AcknowledgementEvidence",
    "AcknowledgementProvider",
    "PostStateEvidence",
    "PostStateProvider",
    "load_fixture_evidence_providers",
    "validate_acknowledgement",
    "validate_post_state",
)

_HEX = frozenset("0123456789abcdef")


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, f"{label} missing")
    return cast("dict[str, object]", value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RolloutViolation(RolloutCode.R_MISSING, f"{label} missing")
    return value


def _digest(value: object, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"{label} invalid")
    return digest


def _time(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise RolloutViolation(RolloutCode.R_NONFINITE, f"{label} invalid")
    return float(value)


def _body(value: object, label: str) -> BodyDegrees:
    if not isinstance(value, list):
        raise RolloutViolation(RolloutCode.R_ACK_MISMATCH, f"{label} invalid")
    values = cast("list[object]", value)
    if len(values) != 5:
        raise RolloutViolation(RolloutCode.R_ACK_MISMATCH, f"{label} invalid")
    degrees = tuple(_time(item, label) for item in values)
    return cast("BodyDegrees", degrees)


@dataclass(frozen=True, slots=True)
class AcknowledgementEvidence:
    """Exact provider acceptance evidence, independent of the write transport."""

    command_id: str
    proposal_hash: str
    observed_at: float
    provider_digest: str
    accepted_body_degrees: BodyDegrees
    digest: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AcknowledgementEvidence:
        """Strictly parse one machine-provided acknowledgement record."""
        expected = {
            "command_id",
            "proposal_hash",
            "observed_at",
            "provider_digest",
            "accepted_body_degrees",
            "digest",
        }
        if set(value) != expected:
            raise RolloutViolation(RolloutCode.R_MISSING, "acknowledgement fields invalid")
        return cls(
            _text(value["command_id"], "ack command_id"),
            _digest(value["proposal_hash"], "ack proposal_hash"),
            _time(value["observed_at"], "ack observed_at"),
            _digest(value["provider_digest"], "ack provider_digest"),
            _body(value["accepted_body_degrees"], "ack body"),
            _digest(value["digest"], "ack digest"),
        )

    def content(self) -> dict[str, object]:
        return {
            "accepted_body_degrees": list(self.accepted_body_degrees),
            "command_id": self.command_id,
            "observed_at": self.observed_at,
            "proposal_hash": self.proposal_hash,
            "provider_digest": self.provider_digest,
        }


@dataclass(frozen=True, slots=True)
class PostStateEvidence:
    """Fresh readback sample acquired after provider acknowledgement."""

    command_id: str
    acknowledgement_digest: str
    created_at: float
    sample_digest: str
    frame_digest: str
    body_degrees: BodyDegrees
    digest: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PostStateEvidence:
        """Strictly parse one synchronized post-command readback record."""
        expected = {
            "command_id",
            "acknowledgement_digest",
            "created_at",
            "sample_digest",
            "frame_digest",
            "body_degrees",
            "digest",
        }
        if set(value) != expected:
            raise RolloutViolation(RolloutCode.R_MISSING, "post-state fields invalid")
        return cls(
            _text(value["command_id"], "post command_id"),
            _digest(value["acknowledgement_digest"], "post acknowledgement_digest"),
            _time(value["created_at"], "post created_at"),
            _digest(value["sample_digest"], "post sample_digest"),
            _digest(value["frame_digest"], "post frame_digest"),
            _body(value["body_degrees"], "post body"),
            _digest(value["digest"], "post digest"),
        )

    def content(self) -> dict[str, object]:
        return {
            "acknowledgement_digest": self.acknowledgement_digest,
            "body_degrees": list(self.body_degrees),
            "command_id": self.command_id,
            "created_at": self.created_at,
            "frame_digest": self.frame_digest,
            "sample_digest": self.sample_digest,
        }


class AcknowledgementProvider(Protocol):
    def acknowledge(self) -> AcknowledgementEvidence:
        """Return exact provider evidence after the one write attempt."""
        ...


class PostStateProvider(Protocol):
    def read_post_state(self) -> PostStateEvidence:
        """Return a newly acquired synchronized physical state."""
        ...


@dataclass(frozen=True, slots=True)
class _FixedAcknowledgementProvider:
    evidence: AcknowledgementEvidence

    def acknowledge(self) -> AcknowledgementEvidence:
        return self.evidence


@dataclass(frozen=True, slots=True)
class _FixedPostStateProvider:
    evidence: PostStateEvidence

    def read_post_state(self) -> PostStateEvidence:
        return self.evidence


def load_fixture_evidence_providers(
    fixture_path: Path,
) -> tuple[AcknowledgementProvider, PostStateProvider, PhysicalIKProposal]:
    """Load fixture bytes into the same typed provider contract used by execution."""
    try:
        raw = json.loads((fixture_path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, "single-step fixture missing") from exc
    manifest = _mapping(raw, "single-step fixture")
    if set(manifest) != {"mode", "proposal", "acknowledgement", "post_state"}:
        raise RolloutViolation(RolloutCode.R_MISSING, "single-step fixture fields invalid")
    if manifest["mode"] != "fixture":
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "fixture mode invalid")
    proposal_document = _mapping(manifest["proposal"], "physical IK proposal")
    declared = _digest(proposal_document.get("proposal_hash"), "proposal hash")
    unhashed = {key: value for key, value in proposal_document.items() if key != "proposal_hash"}
    if physical_ik_proposal_hash(unhashed) != declared:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "proposal content hash")
    proposal = parse_physical_ik_proposal(unhashed, declared_hash=declared)
    ack = AcknowledgementEvidence.from_mapping(
        _mapping(manifest["acknowledgement"], "acknowledgement")
    )
    post = PostStateEvidence.from_mapping(_mapping(manifest["post_state"], "post state"))
    return _FixedAcknowledgementProvider(ack), _FixedPostStateProvider(post), proposal


def validate_acknowledgement(
    evidence: AcknowledgementEvidence,
    *,
    command_id: str,
    proposal_hash: str,
    body_degrees: BodyDegrees,
    newer_than: float,
) -> None:
    """Reject forged, stale, modified, or wrong-command provider evidence."""
    if canonical_hash(evidence.content()) != evidence.digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "acknowledgement digest mismatch")
    if evidence.observed_at <= newer_than:
        raise RolloutViolation(RolloutCode.R_ACK_TIMEOUT, "acknowledgement is not newer")
    if (
        evidence.command_id != command_id
        or evidence.proposal_hash != proposal_hash
        or evidence.accepted_body_degrees != body_degrees
    ):
        raise RolloutViolation(RolloutCode.R_ACK_MISMATCH, "acknowledgement binding mismatch")


def validate_post_state(
    evidence: PostStateEvidence,
    *,
    command_id: str,
    acknowledgement: AcknowledgementEvidence,
    pre_sample_digests: frozenset[str],
) -> None:
    """Reject stale, defaulted, echoed, or content-drifted readback evidence."""
    if not math.isfinite(evidence.created_at) or any(
        not math.isfinite(degree) for degree in evidence.body_degrees
    ):
        raise RolloutViolation(
            RolloutCode.F_POST_STATE_INVALID,
            "post-state timestamp or measured joint is nonfinite",
        )
    if canonical_hash(evidence.content()) != evidence.digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "post-state digest mismatch")
    if evidence.created_at <= acknowledgement.observed_at:
        raise RolloutViolation(RolloutCode.R_POST_STATE_STALE, "post-state is not newer")
    if (
        evidence.command_id != command_id
        or evidence.acknowledgement_digest != acknowledgement.digest
    ):
        raise RolloutViolation(RolloutCode.R_POST_STATE_MISMATCH, "post-state binding mismatch")
    if evidence.sample_digest in pre_sample_digests or evidence.frame_digest in pre_sample_digests:
        raise RolloutViolation(
            RolloutCode.R_DUPLICATE_SAMPLE, "post-state reuses pre-state evidence"
        )
    if evidence.sample_digest in {evidence.frame_digest, acknowledgement.provider_digest}:
        raise RolloutViolation(RolloutCode.R_POST_STATE_MISMATCH, "post-state is fixture echo")
