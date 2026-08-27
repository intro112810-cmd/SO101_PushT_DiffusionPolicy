"""Production read-only acquisition authority and actuation isolation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from live_capture_process_fakes import FakeProviderRuntime
from read_only_authority_fakes import fixture_runtime_preflight
from so101_pusht_benchmark.sim_to_real.policy_approval import (
    ProductionTrustStore,
    RsaPkcs1v15Sha256Anchor,
)
from so101_pusht_benchmark.sim_to_real.arming import ArmingCheckInput, check_arming
from so101_pusht_benchmark.sim_to_real.bounded_authorization import (
    load_bounded_authorization,
)
from so101_pusht_benchmark.sim_to_real.live_capture import capture_live_samples, live_receipt
from so101_pusht_benchmark.sim_to_real.live_capture_types import (
    AdapterIdentity,
    LiveCaptureConfiguration,
    LiveCaptureProviders,
    LiveCaptureRequest,
)
from so101_pusht_benchmark.sim_to_real.policy_parser import require_production_policy
from so101_pusht_benchmark.sim_to_real.read_only_authority import (
    AUTHORITY_SCHEME,
    AUTHORITY_SCHEMA,
    canonical_authority_bytes,
    load_read_only_acquisition_authority,
    path_metadata_digest,
)
from so101_pusht_benchmark.sim_to_real.read_only_authority_builder import (
    validate_provider_semantic_digest,
    validate_source_lineage_digest,
)
from so101_pusht_benchmark.sim_to_real.read_only_authority_runtime import (
    observe_authority_runtime,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.single_step_authorization import (
    load_single_step_authorization,
)
from so101_pusht_benchmark.sim_to_real.supervisor import RolloutSupervisor, SupervisorEvidence
from test_live_sample_capture import (
    FakeCamera,
    FakeJoint,
    camera_reads,
    joint_reads,
)
from so101_pusht_benchmark.sim_to_real.rsa_signing import (
    generate_rsa_private_key,
    public_key_from_private,
    rsa_pkcs1v15_sha256_sign,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)
SOURCE_LINEAGE = "798e982933901ef63d3b2e20fe2a489a23aa67a863340ff6f8bf580532a38850"
PROVIDER = "b1eccc5cbf2c6fd2bb497a40a7070e9ae14ffad2ab7267c1077e804d33629f0a"
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


def test_builder_requires_explicit_source_and_provider_semantic_digests() -> None:
    assert validate_source_lineage_digest(SOURCE_LINEAGE) == SOURCE_LINEAGE
    assert validate_provider_semantic_digest(PROVIDER) == PROVIDER
    for invalid in ("", "0" * 63, "g" * 64):
        with pytest.raises(ValueError, match="source lineage"):
            validate_source_lineage_digest(invalid)
    with pytest.raises(ValueError, match="provider semantic"):
        validate_provider_semantic_digest("0" * 64)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    profile = tmp_path / "profile.yaml"
    calibration = tmp_path / "calibration.json"
    camera = tmp_path / "camera-device"
    follower = tmp_path / "follower-device"
    profile.write_text("profile\n", encoding="utf-8")
    calibration.write_text("calibration\n", encoding="utf-8")
    camera.write_text("camera metadata\n", encoding="utf-8")
    follower.write_text("follower metadata\n", encoding="utf-8")
    return profile, calibration, camera, follower


def _signed_authority(
    tmp_path: Path,
    *,
    omit_threshold: str | None = None,
    runtime_mutation: tuple[str, object] | None = None,
) -> tuple[Path, Path, ProductionTrustStore]:
    profile, calibration, camera, follower = _inputs(tmp_path)
    private = generate_rsa_private_key()
    public = public_key_from_private(private)
    signer = hashlib.sha256(public).hexdigest()
    document: dict[str, object] = {
        "schema": AUTHORITY_SCHEMA,
        "authority_version": 1,
        "authority_id": "readonly-evidence-20260824",
        "artifact_scope": "read_only_evidence_acquisition",
        "approved_by": signer,
        "approved_at": "2026-08-24T12:00:00.000000Z",
        "valid_from": "2026-08-24T12:00:00.000000Z",
        "expires_at": "2026-08-25T12:00:00.000000Z",
        "source_lineage_authority_digest": SOURCE_LINEAGE,
        "provider_digest": PROVIDER,
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
            "sample_max_age_seconds": 0.2,
            "sample_max_skew_seconds": 0.04,
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
        "trust_anchor_sha256": hashlib.sha256(public).hexdigest(),
    }
    if omit_threshold is not None:
        cast("dict[str, object]", document["thresholds"]).pop(omit_threshold)
    if runtime_mutation is not None:
        key, value = runtime_mutation
        cast("dict[str, object]", document["runtime"])[key] = value
    content = canonical_authority_bytes(document)
    document["authority_digest"] = hashlib.sha256(content).hexdigest()
    encoded = canonical_authority_bytes(document)
    authority = tmp_path / "authority.json"
    signature = tmp_path / "authority.sig"
    authority.write_bytes(encoded)
    signature.write_bytes(rsa_pkcs1v15_sha256_sign(private, encoded))
    anchor = RsaPkcs1v15Sha256Anchor(public)
    return authority, signature, ProductionTrustStore.from_owner_anchors((anchor,))


@pytest.mark.parametrize(
    "field",
    [
        "camera_readiness_timeout_seconds",
        "joint_connect_timeout_seconds",
        "sample_pair_completion_timeout_seconds",
        "shutdown_grace_seconds",
        "camera_priming_frame_count",
        "accepted_sample_pair_count",
    ],
)
def test_signed_authority_missing_liveness_field_rejects(
    tmp_path: Path,
    field: str,
) -> None:
    path, signature, trust = _signed_authority(tmp_path, omit_threshold=field)

    with pytest.raises(RolloutViolation) as caught:
        load_read_only_acquisition_authority(
            path,
            signature_path=signature,
            trust_store=trust,
            now=NOW,
        )

    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED


@pytest.mark.parametrize(
    "mutation",
    [
        ("pyserial_version", "3.4"),
        ("feetech_servo_sdk_version", "1.0.1"),
        ("scservo_sdk_origin_sha256", "0" * 64),
    ],
)
def test_resigned_runtime_drift_rejects(
    tmp_path: Path,
    mutation: tuple[str, object],
) -> None:
    path, signature, trust = _signed_authority(tmp_path, runtime_mutation=mutation)

    with pytest.raises(RolloutViolation) as caught:
        load_read_only_acquisition_authority(
            path,
            signature_path=signature,
            trust_store=trust,
            now=NOW,
        )

    assert caught.value.code is RolloutCode.R_PROVIDER_MISMATCH


def test_exact_read_only_authority_loads_with_24h_window(tmp_path: Path) -> None:
    path, signature, trust = _signed_authority(tmp_path)

    authority = load_read_only_acquisition_authority(
        path, signature_path=signature, trust_store=trust, now=NOW
    )

    assert authority.expires_at - authority.valid_from == timedelta(hours=24)
    assert authority.timing.camera_readiness_timeout_seconds == 5.0
    assert authority.timing.joint_connect_timeout_seconds == 5.0
    assert authority.timing.sample_pair_completion_timeout_seconds == 0.2
    assert authority.timing.shutdown_grace_seconds == 1.0
    assert authority.capture.camera_priming_frame_count == 1
    assert authority.capture.accepted_sample_pair_count == 2
    assert authority.timing.sample_max_age_seconds == 0.2
    assert authority.timing.sample_max_skew_seconds == 0.04
    assert authority.camera.max_reprojection_error_px == 1.5
    assert authority.camera.max_correspondence_error_px == 2.0
    assert authority.camera.min_correspondences == 12
    assert authority.kinematics.max_fk_residual_m == 0.003
    assert authority.provider_digest == PROVIDER
    assert authority.source_lineage_authority_digest == SOURCE_LINEAGE
    assert authority.runtime.feetech_servo_sdk_version == "1.0.0"
    assert authority.runtime.pyserial_version == "3.5"
    assert authority.runtime.scservo_sdk_module == "scservo_sdk"
    assert authority.runtime.scservo_sdk_origin.is_absolute()
    assert authority.follower_permissions == (
        "direct_bus_connect",
        "sync_read:Present_Position",
        "disconnect:disable_torque=false",
    )


@pytest.mark.parametrize("attack", ["expired", "tampered", "scope", "fixture"])
def test_expired_tampered_scope_and_fixture_masquerade_reject(tmp_path: Path, attack: str) -> None:
    path, signature, trust = _signed_authority(tmp_path)
    now = NOW
    if attack != "expired":
        document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        if attack == "tampered":
            cast("dict[str, object]", document["thresholds"])["sample_max_age_seconds"] = 0.3
        elif attack == "scope":
            document["artifact_scope"] = "production"
        else:
            document["scheme"] = "rsa-pkcs1v15-sha256-test-fixture-v1"
        path.write_bytes(canonical_authority_bytes(document))
    else:
        now = NOW + timedelta(days=1)

    with pytest.raises(RolloutViolation) as caught:
        load_read_only_acquisition_authority(
            path, signature_path=signature, trust_store=trust, now=now
        )

    assert caught.value.code in {RolloutCode.R_POLICY_UNAUTHORIZED, RolloutCode.R_HASH_MISMATCH}


def test_live_camera_and_present_position_acquisition_accepts_read_only_authority(
    tmp_path: Path,
) -> None:
    path, signature, trust = _signed_authority(tmp_path)
    authority = load_read_only_acquisition_authority(
        path, signature_path=signature, trust_store=trust, now=NOW
    )
    configuration = LiveCaptureConfiguration(
        authority.profile_path,
        authority.camera_device_path,
        authority.follower_device_path,
        authority.calibration_path,
        authority.camera_width,
        authority.camera_height,
        authority.camera_fps,
        authority.calibration_id,
    )
    camera = FakeCamera(
        [],
        camera_reads(),
        identity=AdapterIdentity(authority.provider_digest, authority.camera_device_digest, None),
    )
    joint = FakeJoint(
        [],
        joint_reads(),
        identity=AdapterIdentity(
            authority.provider_digest,
            authority.follower_device_digest,
            authority.calibration_digest,
        ),
    )
    result = capture_live_samples(
        LiveCaptureRequest(authority, authority, configuration),
        LiveCaptureProviders(
            lambda _configuration: camera,
            lambda _configuration: joint,
            path_metadata_digest,
            lambda profile: hashlib.sha256(profile.read_bytes()).hexdigest(),
            lambda calibration: hashlib.sha256(calibration.read_bytes()).hexdigest(),
            lambda: 1000.1,
            fixture_runtime_preflight,
        ),
        process_runtime=FakeProviderRuntime(),
    )

    receipt = live_receipt(result, authority, authority)
    assert receipt["authority_scope"] == "read_only_evidence_acquisition"
    assert receipt["follower_permissions"] == [
        "direct_bus_connect",
        "sync_read:Present_Position",
        "disconnect:disable_torque=false",
    ]
    assert receipt["motor_writes_performed"] is False
    assert receipt["actuation_performed"] is False


def test_read_only_authority_is_structurally_rejected_as_actuation_policy(
    tmp_path: Path,
) -> None:
    path, signature, trust = _signed_authority(tmp_path)
    authority = load_read_only_acquisition_authority(
        path, signature_path=signature, trust_store=trust, now=NOW
    )

    with pytest.raises(RolloutViolation) as caught:
        require_production_policy(authority)

    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED
    assert not hasattr(authority, "workspace")
    assert not hasattr(authority, "joint_domains")
    assert not hasattr(authority, "single_step")
    assert not hasattr(authority, "bounded_rollout")
    assert not hasattr(authority, "provider")
    assert not hasattr(authority, "operator")


def test_arming_rejects_read_only_authority_before_operational_or_device_access(
    tmp_path: Path,
) -> None:
    path, _signature, _trust = _signed_authority(tmp_path)
    fixtures = ROOT / "tests/fixtures/sim_to_real"

    with pytest.raises(RolloutViolation) as caught:
        check_arming(
            ArmingCheckInput(
                ROOT / "configs/hardware/so101_real_v1.yaml",
                path,
                fixtures / "shadow_campaign.jsonl",
                fixtures / "single_step_authorization.json",
                tmp_path / "must-not-be-read-operational-evidence",
                NOW,
            )
        )

    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED


def test_single_step_and_bounded_parsers_reject_read_only_authority(
    tmp_path: Path,
) -> None:
    path, _signature, _trust = _signed_authority(tmp_path)

    with pytest.raises(RolloutViolation):
        load_single_step_authorization(path, now=NOW)
    with pytest.raises(RolloutViolation):
        load_bounded_authorization(
            path,
            now=NOW,
            single_step_receipt_digest="0" * 64,
        )


def test_writer_supervisor_rejects_read_only_authority_before_writer_construction(
    tmp_path: Path,
) -> None:
    path, signature, trust = _signed_authority(tmp_path)
    authority = load_read_only_acquisition_authority(
        path, signature_path=signature, trust_store=trust, now=NOW
    )
    evidence = cast(
        "SupervisorEvidence",
        cast("object", SimpleNamespace(policy=authority)),
    )

    with pytest.raises(RolloutViolation) as caught:
        RolloutSupervisor(lambda: 0.0).mint(evidence)

    assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED
