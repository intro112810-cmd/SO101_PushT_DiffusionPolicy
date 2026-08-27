"""One-shot secure publication of owner-approved read-only acquisition inputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

from so101_pusht_benchmark.hardware_profile import load_hardware_profile

from .live_capture_provider import LIVE_READ_PROVIDER_DIGEST
from .read_only_authority import (
    AUTHORITY_SCHEME,
    AUTHORITY_SCHEMA,
    canonical_authority_bytes,
    path_metadata_digest,
)
from .read_only_authority_runtime import (
    ObservedAuthorityRuntime,
    observe_authority_runtime,
)
from .read_only_authority_thresholds import AcquisitionThresholdInputs
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
from .secure_io import LeafIdentity, atomic_write_new, unlink_owned_leaf

__all__ = (
    "AcquisitionThresholdInputs",
    "build_and_publish",
    "main",
    "validate_provider_semantic_digest",
    "validate_source_lineage_digest",
)
_AUTHORITY_NAME = "read-only-acquisition-authority.json"
_SIGNATURE_NAME = "read-only-acquisition-authority.sig"
_TRUST_NAME = "owner-trust-anchor.pem"


_FORBIDDEN = [
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


def validate_source_lineage_digest(value: str) -> str:
    """Require the explicitly approved settled source-lineage SHA-256 identity."""
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("source lineage authority digest must be 64 lowercase hex characters")
    return normalized


def validate_provider_semantic_digest(value: str) -> str:
    """Bind signing to the exact semantic identity of the live provider code."""
    normalized = validate_source_lineage_digest(value)
    if normalized != LIVE_READ_PROVIDER_DIGEST:
        raise ValueError("provider semantic digest does not match the production provider")
    return normalized


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _DocumentInputs:
    profile_path: Path
    public_key: bytes
    now: datetime
    source_lineage_authority_digest: str
    provider_semantic_digest: str
    runtime: ObservedAuthorityRuntime
    thresholds: AcquisitionThresholdInputs


def _document(inputs: _DocumentInputs) -> dict[str, object]:
    profile = load_hardware_profile(inputs.profile_path)
    canonical_profile = inputs.profile_path.resolve(strict=True)
    calibration = profile.follower.calibration_file.absolute()
    if not calibration.is_file() or calibration.is_symlink():
        raise ValueError("follower calibration must be an existing regular file")
    expires = inputs.now + timedelta(hours=24)
    signer_id = hashlib.sha256(inputs.public_key).hexdigest()
    document: dict[str, object] = {
        "schema": AUTHORITY_SCHEMA,
        "authority_version": 1,
        "authority_id": (
            f"so101-readonly-evidence-{inputs.now.strftime('%Y%m%dT%H%M%SZ')}-"
            f"{inputs.source_lineage_authority_digest[:8]}"
        ),
        "artifact_scope": "read_only_evidence_acquisition",
        "approved_by": signer_id,
        "approved_at": _timestamp(inputs.now),
        "valid_from": _timestamp(inputs.now),
        "expires_at": _timestamp(expires),
        "source_lineage_authority_digest": inputs.source_lineage_authority_digest,
        "provider_digest": inputs.provider_semantic_digest,
        "runtime": inputs.runtime.as_document(),
        "profile": {
            "canonical_path": str(canonical_profile),
            "content_sha256": _sha256_file(canonical_profile),
        },
        "follower": {
            "device_path": str(profile.follower.port.absolute()),
            "device_identity_digest": path_metadata_digest(profile.follower.port),
            "calibration_id": profile.follower.calibration_id,
            "calibration_path": str(calibration),
            "calibration_sha256": _sha256_file(calibration),
        },
        "camera": {
            "device_path": str(profile.camera.device.absolute()),
            "device_identity_digest": path_metadata_digest(profile.camera.device),
            "width": profile.camera.width,
            "height": profile.camera.height,
            "fps": float(profile.camera.fps),
        },
        "thresholds": {
            "camera_readiness_timeout_seconds": (
                inputs.thresholds.camera_readiness_timeout_seconds
            ),
            "joint_connect_timeout_seconds": inputs.thresholds.joint_connect_timeout_seconds,
            "sample_pair_completion_timeout_seconds": (
                inputs.thresholds.sample_pair_completion_timeout_seconds
            ),
            "shutdown_grace_seconds": inputs.thresholds.shutdown_grace_seconds,
            "camera_priming_frame_count": inputs.thresholds.camera_priming_frame_count,
            "accepted_sample_pair_count": inputs.thresholds.accepted_sample_pair_count,
            "sample_max_age_seconds": inputs.thresholds.sample_max_age_seconds,
            "sample_max_skew_seconds": inputs.thresholds.sample_max_skew_seconds,
            "max_fk_residual_m": inputs.thresholds.max_fk_residual_m,
            "max_reprojection_error_px": inputs.thresholds.max_reprojection_error_px,
            "max_correspondence_error_px": inputs.thresholds.max_correspondence_error_px,
            "min_correspondences": inputs.thresholds.min_correspondences,
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
            "forbidden": _FORBIDDEN,
        },
        "scheme": AUTHORITY_SCHEME,
        "trust_anchor_sha256": signer_id,
    }
    document["authority_digest"] = hashlib.sha256(canonical_authority_bytes(document)).hexdigest()
    return document


def _publish(directory: Path, name: str, content: bytes) -> LeafIdentity:
    location = validate_receipt_identity(locate_receipt_path(directory / name), production=True)
    return atomic_write_new(
        location.resolved.parent,
        location.resolved.name,
        content,
        temporary=f".{name}.authority-{os.getpid()}.tmp",
    )


def build_and_publish(
    profile_path: Path,
    output_dir: Path,
    source_lineage_authority_digest: str,
    provider_semantic_digest: str,
    thresholds: AcquisitionThresholdInputs,
) -> dict[str, object]:
    """Generate, sign, verify-route, and publish with no private-key filesystem bytes."""
    source_lineage = validate_source_lineage_digest(source_lineage_authority_digest)
    provider_digest = validate_provider_semantic_digest(provider_semantic_digest)
    runtime = observe_authority_runtime()
    location = validate_receipt_identity(locate_receipt_path(output_dir), production=True)
    prepare_receipt_directory(location.lexical, production=True)
    private_key = generate_rsa_private_key()
    published: list[LeafIdentity] = []
    try:
        public_key = public_key_from_private(private_key)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        document = _document(
            _DocumentInputs(
                profile_path,
                public_key,
                now,
                source_lineage,
                provider_digest,
                runtime,
                thresholds,
            )
        )
        authority_bytes = canonical_authority_bytes(document)
        signature = rsa_pkcs1v15_sha256_sign(private_key, authority_bytes)
        for name, content in (
            (_TRUST_NAME, public_key),
            (_AUTHORITY_NAME, authority_bytes),
            (_SIGNATURE_NAME, signature),
        ):
            published.append(_publish(location.lexical, name, content))
        validate_receipt_identity(
            locate_receipt_path(location.lexical / _AUTHORITY_NAME), production=True
        )
    except Exception:
        for identity in reversed(published):
            unlink_owned_leaf(identity)
        raise
    finally:
        private_key = b""
    return {
        "authority_path": str(location.lexical / _AUTHORITY_NAME),
        "signature_path": str(location.lexical / _SIGNATURE_NAME),
        "trust_anchor_path": str(location.lexical / _TRUST_NAME),
        "authority_digest": document["authority_digest"],
        "signature_sha256": hashlib.sha256(signature).hexdigest(),
        "trust_anchor_sha256": hashlib.sha256(public_key).hexdigest(),
        "expires_at": document["expires_at"],
        "provider_digest": provider_digest,
        "source_lineage_authority_digest": source_lineage,
        "private_key_persisted": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-lineage-authority-digest", required=True)
    parser.add_argument("--provider-semantic-digest", required=True)
    parser.add_argument("--camera-readiness-timeout-seconds", required=True, type=float)
    parser.add_argument("--joint-connect-timeout-seconds", required=True, type=float)
    parser.add_argument("--sample-pair-completion-timeout-seconds", required=True, type=float)
    parser.add_argument("--shutdown-grace-seconds", required=True, type=float)
    parser.add_argument("--camera-priming-frame-count", required=True, type=int)
    parser.add_argument("--accepted-sample-pair-count", required=True, type=int)
    parser.add_argument("--sample-max-age-seconds", required=True, type=float)
    parser.add_argument("--sample-max-skew-seconds", required=True, type=float)
    parser.add_argument("--max-fk-residual-m", required=True, type=float)
    parser.add_argument("--max-reprojection-error-px", required=True, type=float)
    parser.add_argument("--max-correspondence-error-px", required=True, type=float)
    parser.add_argument("--min-correspondences", required=True, type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = build_and_publish(
            args.profile,
            args.output_dir,
            args.source_lineage_authority_digest,
            args.provider_semantic_digest,
            AcquisitionThresholdInputs(
                args.camera_readiness_timeout_seconds,
                args.joint_connect_timeout_seconds,
                args.sample_pair_completion_timeout_seconds,
                args.shutdown_grace_seconds,
                args.camera_priming_frame_count,
                args.accepted_sample_pair_count,
                args.sample_max_age_seconds,
                args.sample_max_skew_seconds,
                args.max_fk_residual_m,
                args.max_reprojection_error_px,
                args.max_correspondence_error_px,
                args.min_correspondences,
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0
