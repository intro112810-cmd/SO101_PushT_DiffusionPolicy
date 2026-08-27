"""Production bounded authorization binds the exact live cycle provider."""

import hashlib
import json
from datetime import datetime, timezone
from so101_pusht_benchmark.sim_to_real.bounded_authorization import (
    verify_bounded_authorization_document,
)


class Verifier:
    def verify(self, signer_id: str, scheme: str, content: bytes, signature_hex: str) -> bool:
        del signer_id, scheme, content, signature_hex
        return True


def test_production_v2_requires_and_returns_cycle_provider_digest() -> None:
    now = datetime(2026, 8, 26, 1, tzinfo=timezone.utc)
    single = "a" * 64
    content = {
        "schema": "so101-bounded-rollout-authorization-v2",
        "artifact_scope": "production",
        "approved_by": "owner",
        "approved_at": "2026-08-26T00:59:00Z",
        "expires_at": "2026-08-26T01:10:00Z",
        "policy_digest": "b" * 64,
        "single_step_receipt_digest": single,
        "cycle_provider_digest": "c" * 64,
        "max_commands": 2,
        "max_duration_seconds": 30.0,
        "max_path_length_m": 0.2,
        "max_error_count": 1,
        "signature_scheme": "rsa-pkcs1v15-sha256-v1",
        "signer_id": "owner",
        "approval_id": "bounded-real",
    }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    doc = {**content, "digest": hashlib.sha256(encoded).hexdigest(), "binding_signature": "00"}
    auth = verify_bounded_authorization_document(
        doc, now=now, single_step_receipt_digest=single, production_verifier=Verifier()
    )
    assert auth.cycle_provider_digest == "c" * 64
