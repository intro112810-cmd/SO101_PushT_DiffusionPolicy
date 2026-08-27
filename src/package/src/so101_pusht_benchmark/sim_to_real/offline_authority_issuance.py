"""Offline persistent-key issuance for read-only joint corpus authorities."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import cast

import yaml

from .joint_positioning_authority import authority_document, load_joint_positioning_authority
from .policy_approval import ProductionTrustStore, RsaPkcs1v15Sha256Anchor
from .policy_canonical import CanonicalIdentity, canonical_content, content_digest
from .policy_parser import SCHEMA as POLICY_SCHEMA
from .policy_schema import YamlMapping, YamlValue, is_mapping_internal
from .policy_parser import load_production_safety_policy
from .policy_values import parse_thresholds_internal
from .read_only_authority import (
    AUTHORITY_SCHEME,
    AUTHORITY_SCHEMA,
    canonical_authority_bytes,
    load_read_only_acquisition_authority,
)
from .read_only_authority_types import (
    CAMERA_PERMISSIONS,
    FORBIDDEN_CAPABILITIES,
    MANUAL_POSITIONING_FOLLOWER_PERMISSIONS,
    ProductionReadOnlyAcquisitionAuthority,
)
from .receipt_routing import (
    locate_receipt_path,
    prepare_receipt_directory,
    validate_receipt_identity,
)
from .rsa_signing import (
    generate_rsa_private_key,
    public_key_from_private,
    rsa_pkcs1v15_sha256_sign,
)
from .secure_io import atomic_write_new, read_regular_leaf

_PRIVATE_KEY = "owner-signing-private-key.pem"
_TRUST_ANCHOR = "owner-trust-anchor.pem"
_BASE_AUTHORITY = "read-only-acquisition-authority.json"
_BASE_SIGNATURE = "read-only-acquisition-authority.sig"
_POLICY = "joint-corpus-production-policy.yaml"
_POSITIONING = "manual-positioning-authority.json"
_POSITIONING_SIGNATURE = "manual-positioning-authority.sig"
_VALIDATION = "offline-validation.json"


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _publish(path: Path, content: bytes) -> None:
    location = validate_receipt_identity(locate_receipt_path(path), production=True)
    atomic_write_new(
        location.resolved.parent,
        location.resolved.name,
        content,
        temporary=f".{path.name}.offline-authority-{os.getpid()}.tmp",
    )


def _read_production(path: Path) -> bytes:
    location = validate_receipt_identity(locate_receipt_path(path), production=True)
    content, _ = read_regular_leaf(location.resolved.parent, location.resolved.name)
    return content


def load_authenticated_source_template(
    authority_path: Path,
    signature_path: Path,
    trust_store: ProductionTrustStore,
) -> ProductionReadOnlyAcquisitionAuthority:
    """Authenticate a signed historical source at its own approval instant."""
    raw: object = json.loads(_read_production(authority_path))
    if not isinstance(raw, Mapping):
        raise TypeError("source authority must be a mapping")
    source_document = cast("Mapping[str, object]", raw)
    approved_at = source_document.get("approved_at")
    if not isinstance(approved_at, str):
        raise TypeError("source authority approved_at is missing")
    verification_time = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    return load_read_only_acquisition_authority(
        authority_path,
        signature_path=signature_path,
        trust_store=trust_store,
        now=verification_time,
    )


def _runtime_document(base: ProductionReadOnlyAcquisitionAuthority) -> dict[str, object]:
    runtime = base.runtime
    return {
        "feetech_servo_sdk_distribution": runtime.feetech_servo_sdk_distribution,
        "feetech_servo_sdk_version": runtime.feetech_servo_sdk_version,
        "pyserial_distribution": runtime.pyserial_distribution,
        "pyserial_version": runtime.pyserial_version,
        "scservo_sdk_distribution": runtime.scservo_sdk_distribution,
        "scservo_sdk_module": runtime.scservo_sdk_module,
        "scservo_sdk_origin": str(runtime.scservo_sdk_origin),
        "scservo_sdk_origin_sha256": runtime.scservo_sdk_origin_sha256,
    }


def _base_document(
    base: ProductionReadOnlyAcquisitionAuthority,
    *,
    signer: str,
    now: datetime,
    source_lineage_authority_digest: str,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": AUTHORITY_SCHEMA,
        "authority_version": 1,
        "authority_id": f"so101-joint-corpus-readonly-{now.strftime('%Y%m%dT%H%M%SZ')}",
        "artifact_scope": "read_only_evidence_acquisition",
        "approved_by": signer,
        "approved_at": _timestamp(now),
        "valid_from": _timestamp(now),
        "expires_at": _timestamp(now + timedelta(hours=24)),
        "source_lineage_authority_digest": source_lineage_authority_digest,
        "provider_digest": base.provider_digest,
        "runtime": _runtime_document(base),
        "profile": {
            "canonical_path": str(base.profile_path),
            "content_sha256": base.profile_digest,
        },
        "follower": {
            "device_path": str(base.follower_device_path),
            "device_identity_digest": base.follower_device_digest,
            "calibration_id": base.calibration_id,
            "calibration_path": str(base.calibration_path),
            "calibration_sha256": base.calibration_digest,
        },
        "camera": {
            "device_path": str(base.camera_device_path),
            "device_identity_digest": base.camera_device_digest,
            "width": base.camera_width,
            "height": base.camera_height,
            "fps": base.camera_fps,
        },
        "thresholds": {
            "camera_readiness_timeout_seconds": base.timing.camera_readiness_timeout_seconds,
            "joint_connect_timeout_seconds": base.timing.joint_connect_timeout_seconds,
            "sample_pair_completion_timeout_seconds": (
                base.timing.sample_pair_completion_timeout_seconds
            ),
            "shutdown_grace_seconds": base.timing.shutdown_grace_seconds,
            "camera_priming_frame_count": base.capture.camera_priming_frame_count,
            "accepted_sample_pair_count": base.capture.accepted_sample_pair_count,
            "sample_max_age_seconds": base.timing.sample_max_age_seconds,
            "sample_max_skew_seconds": base.timing.sample_max_skew_seconds,
            "max_fk_residual_m": base.kinematics.max_fk_residual_m,
            "max_reprojection_error_px": base.camera.max_reprojection_error_px,
            "max_correspondence_error_px": base.camera.max_correspondence_error_px,
            "min_correspondences": base.camera.min_correspondences,
        },
        "permissions": {
            "camera": list(CAMERA_PERMISSIONS),
            "follower": list(MANUAL_POSITIONING_FOLLOWER_PERMISSIONS),
            "forbidden": list(FORBIDDEN_CAPABILITIES),
        },
        "scheme": AUTHORITY_SCHEME,
        "trust_anchor_sha256": signer,
    }
    document["authority_digest"] = hashlib.sha256(canonical_authority_bytes(document)).hexdigest()
    return document


def _policy_document(
    template: Mapping[str, object],
    base: ProductionReadOnlyAcquisitionAuthority,
    *,
    signer: str,
    private_key: bytes,
    now: datetime,
) -> dict[str, object]:
    thresholds_raw = template.get("thresholds")
    if not isinstance(thresholds_raw, Mapping):
        raise TypeError("policy threshold template is invalid")
    copied: object = json.loads(json.dumps(thresholds_raw))
    if not is_mapping_internal(cast("YamlValue", copied)):
        raise TypeError("policy threshold values are invalid")
    thresholds = cast("YamlMapping", copied)
    timing = cast("YamlMapping", thresholds["timing"])
    timing["sample_max_age_seconds"] = base.timing.sample_max_age_seconds
    timing["sample_max_skew_seconds"] = base.timing.sample_max_skew_seconds
    timing["max_policy_age_seconds"] = 86400.0
    camera = cast("YamlMapping", thresholds["camera"])
    camera["max_reprojection_error_px"] = base.camera.max_reprojection_error_px
    camera["min_correspondences"] = base.camera.min_correspondences
    camera["max_correspondence_error_px"] = base.camera.max_correspondence_error_px
    kinematics = cast("YamlMapping", thresholds["kinematics"])
    kinematics["max_fk_residual_m"] = base.kinematics.max_fk_residual_m
    provisional: YamlMapping = {"thresholds": thresholds}
    parsed_thresholds = parse_thresholds_internal(provisional)
    identity = CanonicalIdentity(
        POLICY_SCHEMA,
        1,
        f"joint-corpus-read-only-{now.strftime('%Y%m%dT%H%M%SZ')}",
        "production",
        signer,
        now,
        now,
        now + timedelta(hours=24),
    )
    content = canonical_content(identity, parsed_thresholds)
    policy_digest = content_digest(content)
    approval_id = f"joint-corpus-policy-{now.strftime('%Y%m%dT%H%M%SZ')}"
    signed = {
        "scheme": AUTHORITY_SCHEME,
        "approval_id": approval_id,
        "signer_id": signer,
        "policy_digest": policy_digest,
    }
    approval_signature = rsa_pkcs1v15_sha256_sign(
        private_key,
        json.dumps(signed, sort_keys=True, separators=(",", ":")).encode(),
    ).hex()
    return {
        "schema": POLICY_SCHEMA,
        "policy_version": 1,
        "policy_id": identity.policy_id,
        "artifact_scope": "production",
        "approval_status": "approved",
        "approved_by": signer,
        "approved_at": _timestamp(now),
        "valid_from": _timestamp(now),
        "expires_at": _timestamp(now + timedelta(hours=24)),
        "canonical_digest": policy_digest,
        "thresholds": thresholds,
        "owner_approval": {**signed, "binding_signature": approval_signature},
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(_read_production(path)).hexdigest()


@dataclass(frozen=True, slots=True)
class OfflineIssuanceRequest:
    source_authority: Path
    source_signature: Path
    source_trust_anchor: Path
    policy_template: Path
    lineage_receipt: Path
    output_dir: Path


def issue_offline(request: OfflineIssuanceRequest) -> dict[str, object]:
    """Issue and independently reload all authorities without device opening."""
    source_authority = request.source_authority
    source_signature = request.source_signature
    policy_template = request.policy_template
    lineage_receipt = request.lineage_receipt
    output_dir = request.output_dir
    source_anchor = RsaPkcs1v15Sha256Anchor.from_pem_file(request.source_trust_anchor)
    source_trust = ProductionTrustStore.from_owner_anchors((source_anchor,))
    base = load_authenticated_source_template(source_authority, source_signature, source_trust)
    lineage = cast("dict[str, object]", json.loads(lineage_receipt.read_text(encoding="utf-8")))
    members = lineage.get("members")
    lineage_digest = lineage.get("authority_digest")
    if (
        lineage.get("valid") is not True
        or not isinstance(lineage_digest, str)
        or len(lineage_digest) != 64
        or not isinstance(members, list)
        or len(cast("list[object]", members)) < 204
    ):
        raise ValueError("exact source lineage receipt is not valid")
    template_raw: object = yaml.safe_load(policy_template.read_text(encoding="utf-8"))
    if not isinstance(template_raw, Mapping):
        raise TypeError("policy threshold template must be a mapping")
    prepare_receipt_directory(output_dir, production=True)
    private_key = generate_rsa_private_key()
    public_key = public_key_from_private(private_key)
    signer = hashlib.sha256(public_key).hexdigest()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    paths = {
        "private_key": output_dir / _PRIVATE_KEY,
        "trust_anchor": output_dir / _TRUST_ANCHOR,
        "base_authority": output_dir / _BASE_AUTHORITY,
        "base_signature": output_dir / _BASE_SIGNATURE,
        "policy": output_dir / _POLICY,
        "positioning_authority": output_dir / _POSITIONING,
        "positioning_signature": output_dir / _POSITIONING_SIGNATURE,
        "validation": output_dir / _VALIDATION,
    }
    base_document = _base_document(
        base,
        signer=signer,
        now=now,
        source_lineage_authority_digest=lineage_digest,
    )
    base_encoded = canonical_authority_bytes(base_document)
    policy_document = _policy_document(
        cast("Mapping[str, object]", template_raw),
        base,
        signer=signer,
        private_key=private_key,
        now=now,
    )
    policy_encoded = yaml.safe_dump(policy_document, sort_keys=False).encode()
    for path, content in (
        (paths["private_key"], private_key),
        (paths["trust_anchor"], public_key),
        (paths["base_authority"], base_encoded),
        (paths["base_signature"], rsa_pkcs1v15_sha256_sign(private_key, base_encoded)),
        (paths["policy"], policy_encoded),
    ):
        _publish(path, content)
    paths["private_key"].resolve(strict=True).chmod(0o600)
    owner_anchor = RsaPkcs1v15Sha256Anchor.from_pem_file(paths["trust_anchor"])
    owner_trust = ProductionTrustStore.from_owner_anchors((owner_anchor,))
    issued_base = load_read_only_acquisition_authority(
        paths["base_authority"],
        signature_path=paths["base_signature"],
        trust_store=owner_trust,
        now=now,
    )
    positioning_document = authority_document(
        issued_base,
        authority_id=f"joint-corpus-positioning-{now.strftime('%Y%m%dT%H%M%SZ')}",
        approved_by=signer,
        valid_from=now,
    )
    positioning_encoded = canonical_authority_bytes(positioning_document)
    _publish(paths["positioning_authority"], positioning_encoded)
    _publish(
        paths["positioning_signature"],
        rsa_pkcs1v15_sha256_sign(private_key, positioning_encoded),
    )
    issued_policy = load_production_safety_policy(paths["policy"], trust_store=owner_trust, now=now)
    issued_positioning = load_joint_positioning_authority(
        paths["positioning_authority"],
        signature_path=paths["positioning_signature"],
        trust_store=owner_trust,
        base=issued_base,
        now=now,
    )
    validation: dict[str, object] = {
        "schema": "so101-offline-joint-corpus-authority-validation-v1",
        "validated_at": _timestamp(now),
        "valid": True,
        "hardware_open_count": 0,
        "motor_write_count": 0,
        "lineage_authority_digest": issued_base.source_lineage_authority_digest,
        "lineage_member_count": len(cast("list[object]", members)),
        "authority_digest": issued_base.canonical_digest,
        "policy_digest": issued_policy.canonical_digest,
        "positioning_authority_digest": issued_positioning.canonical_digest,
        "base_permissions": list(issued_base.follower_permissions),
        "permissions": list(issued_positioning.permissions),
        "forbidden_capabilities": list(issued_base.forbidden_capabilities),
        "valid_from": _timestamp(issued_base.valid_from),
        "expires_at": _timestamp(issued_base.expires_at),
        "timeout_values_preserved": {
            "camera_readiness_timeout_seconds": issued_base.timing.camera_readiness_timeout_seconds,
            "joint_connect_timeout_seconds": issued_base.timing.joint_connect_timeout_seconds,
            "sample_pair_completion_timeout_seconds": (
                issued_base.timing.sample_pair_completion_timeout_seconds
            ),
            "shutdown_grace_seconds": issued_base.timing.shutdown_grace_seconds,
            "sample_max_age_seconds": issued_base.timing.sample_max_age_seconds,
            "sample_max_skew_seconds": issued_base.timing.sample_max_skew_seconds,
        },
        "sha256": (
            hashes := {name: _sha(path) for name, path in paths.items() if name != "validation"}
        ),
        "paths": {name: str(path) for name, path in paths.items()},
        "private_key_mode": oct(paths["private_key"].resolve(strict=True).stat().st_mode & 0o777),
    }
    _publish(paths["validation"], canonical_authority_bytes(validation))
    hashes["validation"] = _sha(paths["validation"])
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-authority", required=True, type=Path)
    parser.add_argument("--source-signature", required=True, type=Path)
    parser.add_argument("--source-trust-anchor", required=True, type=Path)
    parser.add_argument("--policy-template", required=True, type=Path)
    parser.add_argument("--lineage-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = issue_offline(
            OfflineIssuanceRequest(
                args.source_authority,
                args.source_signature,
                args.source_trust_anchor,
                args.policy_template,
                args.lineage_receipt,
                args.output_dir,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
