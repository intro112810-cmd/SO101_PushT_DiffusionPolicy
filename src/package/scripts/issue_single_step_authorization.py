#!/usr/bin/env python3
"""Issue one owner-signed production single-step authorization offline."""

from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import cast
from so101_pusht_benchmark.hardware_profile import load_hardware_profile
from so101_pusht_benchmark.sim_to_real.arming import ArmingCheckInput, check_production_arming
from so101_pusht_benchmark.sim_to_real.arming_evidence import load_operational_evidence
from so101_pusht_benchmark.sim_to_real.ledger_chain import verify_ledger
from so101_pusht_benchmark.sim_to_real.ledger_io import load_ledger_documents
from so101_pusht_benchmark.sim_to_real.policy_approval import (
    ProductionTrustStore,
    RsaPkcs1v15Sha256Anchor,
)
from so101_pusht_benchmark.sim_to_real.policy_parser import load_production_safety_policy
from so101_pusht_benchmark.sim_to_real.receipt_routing import prepare_receipt_directory
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutViolation
from so101_pusht_benchmark.sim_to_real.single_step_authorization_issuance import (
    AuthorizationIssuanceMaterial,
    issue_single_step_authorization,
)


def _package(path: Path) -> tuple[str, str, str]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("execution package must be a mapping")
    doc = cast("dict[str,object]", raw)
    proposal = doc.get("proposal")
    token = doc.get("token")
    if not isinstance(proposal, dict) or not isinstance(token, dict):
        raise TypeError("package proposal/token missing")
    proposal_document = cast("dict[str, object]", proposal)
    token_document = cast("dict[str, object]", token)
    proposal_hash = proposal_document.get("proposal_hash")
    command_id = token_document.get("command_id")
    policy_digest = token_document.get("policy_digest")
    if not all(isinstance(value, str) for value in (proposal_hash, command_id, policy_digest)):
        raise ValueError("package bindings invalid")
    return cast("str", proposal_hash), cast("str", command_id), cast("str", policy_digest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "profile",
        "policy",
        "trust-anchor",
        "private-key",
        "shadow-ledger",
        "operational-evidence",
        "package",
        "output",
    ):
        p.add_argument(f"--{name}", required=True, type=Path)
    p.add_argument("--approval-id", required=True)
    a = p.parse_args()
    now = datetime.now(timezone.utc)
    try:
        profile = load_hardware_profile(a.profile)
        anchor = RsaPkcs1v15Sha256Anchor.from_pem_file(a.trust_anchor)
        trust = ProductionTrustStore.from_owner_anchors((anchor,))
        policy = load_production_safety_policy(a.policy, trust_store=trust, now=now)
        proposal_hash, command_id, token_policy = _package(a.package)
        if (
            profile.policy_digest != policy.canonical_digest
            or token_policy != policy.canonical_digest
        ):
            raise ValueError("profile/package policy binding mismatch")
        records = load_ledger_documents(a.shadow_ledger)
        verify_ledger(records)
        operational = load_operational_evidence(
            a.operational_evidence,
            now=now,
            max_age_seconds=policy.timing.authorization_max_age_seconds,
        )
        private_key = a.private_key.read_bytes()
        expires = min(
            now + timedelta(seconds=policy.timing.authorization_ttl_seconds), policy.expires_at
        )
        material = AuthorizationIssuanceMaterial(
            anchor.signer_id,
            now,
            expires,
            policy.canonical_digest,
            proposal_hash,
            command_id,
            operational.ownership_digest,
            operational.interlock_digest,
            operational.torque_digest,
            hashlib.sha256(a.shadow_ledger.read_bytes()).hexdigest(),
            a.approval_id,
        )
        document = issue_single_step_authorization(material, private_key)
        prepare_receipt_directory(a.output.parent, production=True)
        a.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        check_production_arming(
            ArmingCheckInput(
                a.profile, a.policy, a.shadow_ledger, a.output, a.operational_evidence, now
            ),
            trust,
        )
    except (OSError, RuntimeError, TypeError, ValueError, RolloutViolation) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "authorization": str(a.output),
                "command_id": command_id,
                "proposal_hash": proposal_hash,
                "expires_at": expires.isoformat(),
                "hardware_opened": False,
                "motor_writes": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
