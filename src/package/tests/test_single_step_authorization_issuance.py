"""Owner-key production single-step authorization issuance."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from so101_pusht_benchmark.sim_to_real.policy_approval import (
    ProductionTrustStore,
    RsaPkcs1v15Sha256Anchor,
)
from so101_pusht_benchmark.sim_to_real.rsa_signing import (
    generate_rsa_private_key,
    public_key_from_private,
)
from so101_pusht_benchmark.sim_to_real.single_step_authorization import (
    load_single_step_authorization,
)
from so101_pusht_benchmark.sim_to_real.single_step_authorization_issuance import (
    AuthorizationIssuanceMaterial,
    issue_single_step_authorization,
)


def test_issued_authorization_reloads_through_owner_anchor(tmp_path: Path) -> None:
    key = generate_rsa_private_key()
    public = public_key_from_private(key)
    now = datetime(2026, 8, 26, 1, tzinfo=timezone.utc)
    signer = __import__("hashlib").sha256(public).hexdigest()
    material = AuthorizationIssuanceMaterial(
        signer,
        now,
        now + timedelta(seconds=30),
        "a" * 64,
        "b" * 64,
        "command-real",
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "f" * 64,
        "approval-real",
    )
    document = issue_single_step_authorization(material, key)
    path = tmp_path / "authorization.json"
    path.write_text(__import__("json").dumps(document))
    anchor_path = tmp_path / "anchor.pem"
    anchor_path.write_bytes(public)
    trust = ProductionTrustStore.from_owner_anchors(
        (RsaPkcs1v15Sha256Anchor.from_pem_file(anchor_path),)
    )
    loaded = load_single_step_authorization(path, now=now, production_verifier=trust)
    assert loaded.command_id == "command-real"
    assert loaded.proposal_hash == "b" * 64
    assert loaded.armed_receipt_digest == document["armed_receipt_digest"]
