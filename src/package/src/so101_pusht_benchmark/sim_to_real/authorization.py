"""Proposal-bound, expiring authorization identity for guarded rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import math
import string
from .rollout_codes import RolloutCode, RolloutViolation
from .rollout_identity import BoundaryValue, digest_content

__all__ = (
    "AuthorizationClaim",
    "AuthorizationToken",
    "mint_authorization",
    "verify_authorization",
)
_HEX = frozenset(string.hexdigits.lower())


@dataclass(frozen=True, slots=True)
class AuthorizationClaim:
    """The proposal, policy, command, and expiry bound into one token."""

    proposal_hash: str
    policy_digest: str
    command_id: str
    valid_until: float


@dataclass(frozen=True, slots=True)
class AuthorizationToken:
    """Immutable single-use claim; the supervisor records its consumption."""

    token_id: str
    proposal_hash: str
    policy_digest: str
    command_id: str
    valid_until: float
    digest: str

    def __post_init__(self) -> None:
        """Reject malformed or content-drifted tokens at their trust boundary."""
        verify_authorization(self)


def _sha(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, label)
    return value


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise RolloutViolation(RolloutCode.R_NONFINITE, "authorization valid_until")
    return value


def _content(token_id: str, claim: AuthorizationClaim) -> dict[str, BoundaryValue]:
    return {
        "kind": "rollout_authorization",
        "token_id": token_id,
        "proposal_hash": claim.proposal_hash,
        "policy_digest": claim.policy_digest,
        "command_id": claim.command_id,
        "valid_until": claim.valid_until,
    }


def _claim(token: AuthorizationToken) -> AuthorizationClaim:
    return AuthorizationClaim(
        proposal_hash=token.proposal_hash,
        policy_digest=token.policy_digest,
        command_id=token.command_id,
        valid_until=token.valid_until,
    )


def mint_authorization(claim: AuthorizationClaim) -> AuthorizationToken:
    """Mint deterministic content identity after the supervisor accepts evidence."""
    token_id = digest_content(
        {
            "kind": "rollout_authorization_id",
            "proposal_hash": claim.proposal_hash,
            "policy_digest": claim.policy_digest,
            "command_id": claim.command_id,
            "valid_until": claim.valid_until,
        }
    )
    return AuthorizationToken(
        token_id=token_id,
        proposal_hash=claim.proposal_hash,
        policy_digest=claim.policy_digest,
        command_id=claim.command_id,
        valid_until=claim.valid_until,
        digest=digest_content(_content(token_id, claim)),
    )


def verify_authorization(token: AuthorizationToken) -> None:
    """Verify the token's immutable content binding without consuming it."""
    _sha(token.token_id, "authorization token_id")
    _sha(token.proposal_hash, "authorization proposal_hash")
    _sha(token.policy_digest, "authorization policy_digest")
    _sha(token.digest, "authorization digest")
    _finite(token.valid_until)
    if not token.command_id:
        raise RolloutViolation(RolloutCode.R_MISSING, "authorization command_id")
    if digest_content(_content(token.token_id, _claim(token))) != token.digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "authorization content")
