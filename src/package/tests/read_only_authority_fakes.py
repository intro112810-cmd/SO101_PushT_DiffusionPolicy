"""Temporary signed read-only authorities used only by injected tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path

from so101_pusht_benchmark.sim_to_real.policy_approval import (
    ProductionTrustStore,
    RsaPkcs1v15Sha256Anchor,
)
from so101_pusht_benchmark.sim_to_real.read_only_authority import (
    AUTHORITY_SCHEME,
    AUTHORITY_SCHEMA,
    ProductionReadOnlyAcquisitionAuthority,
    canonical_authority_bytes,
    load_read_only_acquisition_authority,
    path_metadata_digest,
)
from so101_pusht_benchmark.sim_to_real.live_capture_runtime import RuntimeDependencyReceipt
from so101_pusht_benchmark.sim_to_real.read_only_authority_runtime import (
    observe_authority_runtime,
)
from so101_pusht_benchmark.sim_to_real.rsa_signing import (
    generate_rsa_private_key,
    public_key_from_private,
    rsa_pkcs1v15_sha256_sign,
)

NOW = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)


def fixture_runtime_preflight() -> RuntimeDependencyReceipt:
    """Return test-only runtime identity without importing a hardware SDK."""
    return RuntimeDependencyReceipt(
        "feetech-servo-sdk",
        "1.0.0",
        "scservo_sdk",
        Path(__file__).resolve(),
    )


FORBIDDEN = [
    "Goal_Position",
    "sync_write",
    "torque_write",
    "configuration_write",
    "calibration_write",
    "SOFollower.connect",
    "SOFollower.configure",
    "SOFollower.calibrate",
    "SOFollower.send_action",
    "single_step",
    "bounded_rollout",
    "arming",
]


def signed_test_authority(
    root: Path,
    *,
    provider_digest: str,
    sample_max_age_seconds: float = 0.2,
    sample_max_skew_seconds: float = 0.04,
) -> ProductionReadOnlyAcquisitionAuthority:
    """Create and verify one test-only key/signature without persistent private bytes."""
    profile = root / "profile.yaml"
    calibration = root / "calibration.json"
    camera = root / "camera-device"
    follower = root / "follower-device"
    if not profile.exists():
        profile.write_text("profile\n", encoding="utf-8")
    if not calibration.exists():
        calibration.write_text("calibration\n", encoding="utf-8")
    if not camera.exists():
        camera.write_text("camera metadata\n", encoding="utf-8")
    if not follower.exists():
        follower.write_text("follower metadata\n", encoding="utf-8")
    private = generate_rsa_private_key()
    public = public_key_from_private(private)
    signer = hashlib.sha256(public).hexdigest()
    document: dict[str, object] = {
        "schema": AUTHORITY_SCHEMA,
        "authority_version": 1,
        "authority_id": "test-only-readonly-evidence",
        "artifact_scope": "read_only_evidence_acquisition",
        "approved_by": signer,
        "approved_at": NOW.isoformat(),
        "valid_from": NOW.isoformat(),
        "expires_at": (NOW + timedelta(hours=24)).isoformat(),
        "source_lineage_authority_digest": "a" * 64,
        "provider_digest": provider_digest,
        "runtime": observe_authority_runtime().as_document(),
        "profile": {
            "canonical_path": str(profile.resolve()),
            "content_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        },
        "follower": {
            "device_path": str(follower.resolve()),
            "device_identity_digest": path_metadata_digest(follower),
            "calibration_id": "follower-01",
            "calibration_path": str(calibration.resolve()),
            "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        },
        "camera": {
            "device_path": str(camera.resolve()),
            "device_identity_digest": path_metadata_digest(camera),
            "width": 640,
            "height": 480,
            "fps": 30.0,
        },
        "thresholds": {
            "camera_readiness_timeout_seconds": 5.0,
            "joint_connect_timeout_seconds": 5.0,
            "sample_pair_completion_timeout_seconds": 0.2,
            "shutdown_grace_seconds": 1.0,
            "camera_priming_frame_count": 1,
            "accepted_sample_pair_count": 2,
            "sample_max_age_seconds": sample_max_age_seconds,
            "sample_max_skew_seconds": sample_max_skew_seconds,
            "max_fk_residual_m": 0.003,
            "max_reprojection_error_px": 1.5,
            "max_correspondence_error_px": 2.0,
            "min_correspondences": 12,
        },
        "permissions": {
            "camera": [
                "open_existing_capture",
                "observe_properties",
                "read_frames",
                "release_capture",
            ],
            "follower": [
                "direct_bus_connect",
                "sync_read:Present_Position",
                "disconnect:disable_torque=false",
            ],
            "forbidden": FORBIDDEN,
        },
        "scheme": AUTHORITY_SCHEME,
        "trust_anchor_sha256": signer,
    }
    document["authority_digest"] = hashlib.sha256(canonical_authority_bytes(document)).hexdigest()
    encoded = canonical_authority_bytes(document)
    authority_path = root / "authority.json"
    signature_path = root / "authority.sig"
    authority_path.write_bytes(encoded)
    signature_path.write_bytes(rsa_pkcs1v15_sha256_sign(private, encoded))
    trust = ProductionTrustStore.from_owner_anchors((RsaPkcs1v15Sha256Anchor(public),))
    return load_read_only_acquisition_authority(
        authority_path,
        signature_path=signature_path,
        trust_store=trust,
        now=NOW,
    )
