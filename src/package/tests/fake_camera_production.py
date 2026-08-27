from __future__ import annotations

import hashlib

from so101_pusht_benchmark.sim_to_real import policy_approval
from so101_pusht_benchmark.sim_to_real.policy_approval import ProductionTrustStore

OWNER = "camera-fixture-owner@example.invalid"
SCHEME = "fake-production-sha256-v1"
_PREFIX = b"fake-camera-production-anchor-v1:"


def fake_signature(content: bytes) -> str:
    return hashlib.sha256(_PREFIX + content).hexdigest()


class FakeCameraProductionAnchor:
    def verify(self, signer_id: str, scheme: str, content: bytes, signature_hex: str) -> bool:
        return signer_id == OWNER and scheme == SCHEME and signature_hex == fake_signature(content)


def fake_production_trust_store() -> ProductionTrustStore:
    seal = policy_approval.__dict__["_PRODUCTION_STORE_SEAL"]
    return ProductionTrustStore(seal, (FakeCameraProductionAnchor(),))
