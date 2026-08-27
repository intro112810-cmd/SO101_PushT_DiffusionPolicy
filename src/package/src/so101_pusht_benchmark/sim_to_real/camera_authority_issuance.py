"""Offline owner-key issuance for one exact physical camera corpus."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import cast

from .camera_authority import verify_production_camera_authority
from .live_capture_identity import load_approved_live_identity
from .policy_approval import ProductionTrustStore, RsaPkcs1v15Sha256Anchor
from .policy_parser import load_production_safety_policy
from .read_only_authority import load_read_only_acquisition_authority
from .rsa_signing import public_key_from_private, rsa_pkcs1v15_sha256_sign
from .secure_io import atomic_write_new

_SCHEME = "rsa-pkcs1v15-sha256-v1"


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _mapping(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a mapping")
    return cast("dict[str, object]", value)


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} is not a digest")
    bytes.fromhex(value)
    return value


def _publish(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_new(path.parent.resolve(), path.name, content, temporary=f".{path.name}.tmp")


@dataclass(frozen=True, slots=True)
class CameraAuthorityIssuanceRequest:
    base_authority: Path
    base_signature: Path
    policy: Path
    trust_anchor: Path
    private_key: Path
    corpus: Path
    output_dir: Path
    approval_id: str


def issue_camera_authorities(request: CameraAuthorityIssuanceRequest) -> dict[str, object]:
    """Sign exact provider identity and corpus bindings without opening hardware."""
    anchor = RsaPkcs1v15Sha256Anchor.from_pem_file(request.trust_anchor)
    trust = ProductionTrustStore.from_owner_anchors((anchor,))
    base = load_read_only_acquisition_authority(
        request.base_authority, signature_path=request.base_signature, trust_store=trust
    )
    policy = load_production_safety_policy(request.policy, trust_store=trust)
    private_key = request.private_key.read_bytes()
    signer = hashlib.sha256(public_key_from_private(private_key)).hexdigest()
    if signer != anchor.signer_id or signer != base.approved_by or signer != policy.approved_by:
        raise ValueError("private key, authority, policy, and trust anchor owner differ")
    corpus = _mapping(request.corpus)
    corpus_digest = _digest(corpus.get("camera_digest"), "camera digest")
    orientation = _digest(corpus.get("orientation_hash"), "orientation digest")
    identity_content: dict[str, object] = {
        "schema": "live-read-identity-v1",
        "artifact_scope": "production",
        "provider_digest": base.provider_digest,
        "profile_digest": base.profile_digest,
        "camera_device_digest": base.camera_device_digest,
        "follower_device_digest": base.follower_device_digest,
        "calibration_digest": base.calibration_digest,
        "camera_width": base.camera_width,
        "camera_height": base.camera_height,
        "camera_fps": base.camera_fps,
        "approved_by": signer,
        "approval_id": request.approval_id + "-identity",
    }
    identity_digest = hashlib.sha256(_canonical(identity_content)).hexdigest()
    identity_binding = _canonical(
        {
            "approval_id": identity_content["approval_id"],
            "identity_digest": identity_digest,
            "schema": identity_content["schema"],
            "signer_id": signer,
        }
    )
    identity = {
        **identity_content,
        "identity_digest": identity_digest,
        "scheme": _SCHEME,
        "binding_signature": rsa_pkcs1v15_sha256_sign(private_key, identity_binding).hex(),
    }
    authority_content: dict[str, object] = {
        "schema": "camera-corpus-authority-v1",
        "artifact_scope": "production",
        "approved_by": signer,
        "approval_id": request.approval_id + "-corpus",
        "scheme": _SCHEME,
        "corpus_digest": corpus_digest,
        "live_identity_digest": identity_digest,
        "provider_digest": base.provider_digest,
        "profile_digest": base.profile_digest,
        "camera_device_digest": base.camera_device_digest,
        "calibration_digest": base.calibration_digest,
        "orientation_digest": orientation,
    }
    authority = {
        **authority_content,
        "binding_signature": rsa_pkcs1v15_sha256_sign(
            private_key, _canonical(authority_content)
        ).hex(),
    }
    identity_path = request.output_dir / "live-identity.json"
    authority_path = request.output_dir / "camera-corpus-authority.json"
    _publish(identity_path, _canonical(identity) + b"\n")
    _publish(authority_path, _canonical(authority) + b"\n")
    loaded_identity = load_approved_live_identity(identity_path, trust_store=trust)
    verified_authority, _ = verify_production_camera_authority(
        corpus,
        authority_path=authority_path,
        identity_path=identity_path,
        policy=policy,
        trust_store=trust,
    )
    return {
        "identity": str(identity_path),
        "identity_digest": loaded_identity.identity_digest,
        "camera_authority": str(authority_path),
        "corpus_digest": verified_authority.corpus_digest,
        "hardware_open_count": 0,
        "motor_write_count": 0,
    }
