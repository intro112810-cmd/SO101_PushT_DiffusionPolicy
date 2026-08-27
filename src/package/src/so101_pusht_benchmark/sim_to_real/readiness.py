"""Content-addressed rollout readiness checks.

Boolean profile flags never promote a physical diagnostic. Bound SHA-256
digests on the profile must match the inference receipt before readiness
can clear.
"""

from __future__ import annotations

from collections.abc import Mapping

DIGEST_LENGTH = 64

_MISSING = {
    "lineage_digest": "lineage evidence is missing its content digest",
    "policy_digest": "policy evidence is missing its content digest",
    "camera_registration_digest": "camera registration evidence is missing its receipt hash",
    "joint_equivalence_digest": "joint mapping evidence is missing a valid receipt status",
}


def is_sha256_digest(value: object) -> bool:
    """Return True when value is a lowercase 64-character hex digest."""
    return (
        isinstance(value, str)
        and len(value) == DIGEST_LENGTH
        and set(value) <= set("0123456789abcdef")
    )


def _mismatch(label: str) -> str:
    return f"{label} does not match bound evidence"


def bound_digest_blockers(
    profile_digest: str,
    receipt: Mapping[str, object],
    key: str,
) -> tuple[str, ...]:
    """Require a matching SHA-256 on both the profile and the receipt."""
    receipt_digest = receipt.get(key)
    if not is_sha256_digest(profile_digest) or not is_sha256_digest(receipt_digest):
        return (_MISSING[key],)
    if profile_digest != receipt_digest:
        return (_mismatch(key),)
    return ()


def joint_mapping_blockers(
    profile_digest: str,
    receipt: Mapping[str, object],
) -> tuple[str, ...]:
    """Require a valid joint-mapping receipt plus a matching equivalence digest."""
    blockers = list(bound_digest_blockers(profile_digest, receipt, "joint_equivalence_digest"))
    status = receipt.get("joint_mapping_receipt_status")
    valid_flag = receipt.get("joint_mapping_valid")
    if status != "valid" and valid_flag is not True:
        missing = _MISSING["joint_equivalence_digest"]
        if missing not in blockers:
            blockers.append(missing)
    return tuple(blockers)


def evidence_is_fully_bound(blockers: tuple[str, ...]) -> bool:
    """Return True when no content-addressed evidence message remains."""
    missing = set(_MISSING.values())
    return not any(
        item in missing or item.endswith("does not match bound evidence") for item in blockers
    )
