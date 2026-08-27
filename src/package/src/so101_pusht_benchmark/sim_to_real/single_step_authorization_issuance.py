"""Offline issuance of one exact production single-step authorization."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from .ledger_chain import canonical_hash
from .rsa_signing import public_key_from_private, rsa_pkcs1v15_sha256_sign


@dataclass(frozen=True, slots=True)
class AuthorizationIssuanceMaterial:
    approved_by: str
    approved_at: datetime
    expires_at: datetime
    policy_digest: str
    proposal_hash: str
    command_id: str
    ownership_digest: str
    interlock_digest: str
    torque_digest: str
    shadow_ledger_digest: str
    approval_id: str


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def issue_single_step_authorization(
    material: AuthorizationIssuanceMaterial, private_key: bytes
) -> dict[str, object]:
    """Sign one command after exact proposal, operational, and shadow binding."""
    signer = hashlib.sha256(public_key_from_private(private_key)).hexdigest()
    if signer != material.approved_by:
        raise ValueError("private key and approved owner differ")
    armed_content = {
        "armed": True,
        "command_id": material.command_id,
        "interlock_digest": material.interlock_digest,
        "motor_writes_performed": False,
        "ownership_digest": material.ownership_digest,
        "policy_digest": material.policy_digest,
        "proposal_hash": material.proposal_hash,
        "shadow_ledger_digest": material.shadow_ledger_digest,
        "torque_digest": material.torque_digest,
    }
    content: dict[str, object] = {
        "schema": "so101-single-step-authorization-v1",
        "artifact_scope": "production",
        "approved_by": signer,
        "approved_at": _timestamp(material.approved_at),
        "expires_at": _timestamp(material.expires_at),
        "policy_digest": material.policy_digest,
        "proposal_hash": material.proposal_hash,
        "command_id": material.command_id,
        "command_budget": 1,
        "ownership_digest": material.ownership_digest,
        "interlock_digest": material.interlock_digest,
        "torque_digest": material.torque_digest,
        "armed_receipt_digest": canonical_hash(armed_content),
        "signature_scheme": "rsa-pkcs1v15-sha256-v1",
        "signer_id": signer,
        "approval_id": material.approval_id,
    }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return {
        **content,
        "digest": hashlib.sha256(encoded).hexdigest(),
        "binding_signature": rsa_pkcs1v15_sha256_sign(private_key, encoded).hex(),
    }
