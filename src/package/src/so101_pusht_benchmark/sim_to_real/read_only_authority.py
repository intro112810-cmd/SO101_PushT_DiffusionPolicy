"""Authenticate and bind one signed production read-only acquisition authority."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import cast

from .policy_approval import ProductionTrustStore
from .read_only_authority_io import (
    sha256_digest,
    parse_mapping,
    positive_number,
    positive_integer,
    read_regular,
    required_text,
    parse_timestamp,
    verify_current_bindings,
    authority_violation,
    path_metadata_digest,
)
from .read_only_authority_runtime import RUNTIME_FIELDS, observe_authority_runtime
from .read_only_authority_types import (
    AUTHORITY_SCHEMA,
    AUTHORITY_SCHEME,
    CameraReadinessTimeoutSeconds,
    JointConnectTimeoutSeconds,
    PairCompletionTimeoutSeconds,
    ProductionReadOnlyAcquisitionAuthority,
    ReadOnlyCameraPolicy,
    ReadOnlyCapturePolicy,
    ReadOnlyKinematicsPolicy,
    ReadOnlyRuntimePolicy,
    ReadOnlyTimingPolicy,
    ShutdownGraceSeconds,
    CAMERA_PERMISSIONS,
    FOLLOWER_PERMISSIONS,
    MANUAL_POSITIONING_FOLLOWER_PERMISSIONS,
    FORBIDDEN_CAPABILITIES,
    READ_ONLY_CONSTRUCTION_SEAL,
    AUTHORITY_SCOPE,
    AUTHORITY_FIELDS,
    canonical_authority_bytes,
    require_read_only_acquisition_authority,
)
from .rollout_codes import RolloutCode

__all__ = (
    "AUTHORITY_SCHEMA",
    "AUTHORITY_SCHEME",
    "CameraReadinessTimeoutSeconds",
    "JointConnectTimeoutSeconds",
    "PairCompletionTimeoutSeconds",
    "ProductionReadOnlyAcquisitionAuthority",
    "ReadOnlyTimingPolicy",
    "ShutdownGraceSeconds",
    "canonical_authority_bytes",
    "load_read_only_acquisition_authority",
    "path_metadata_digest",
    "require_read_only_acquisition_authority",
)


def load_read_only_acquisition_authority(
    path: Path,
    *,
    signature_path: Path,
    trust_store: ProductionTrustStore,
    now: datetime | None = None,
) -> ProductionReadOnlyAcquisitionAuthority:
    """Independently parse, authenticate, and bind one read-only authority."""
    if type(trust_store) is not ProductionTrustStore or not trust_store.is_governed():
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "governed trust store required"
        )
    encoded = read_regular(path, "read-only acquisition authority")
    try:
        value: object = json.loads(encoded)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "authority JSON is invalid"
        ) from exc
    raw = parse_mapping(value, AUTHORITY_FIELDS, "authority")
    if canonical_authority_bytes(raw) != encoded:
        raise authority_violation(RolloutCode.R_HASH_MISMATCH, "authority is not canonical JSON")
    if (
        raw["schema"] != AUTHORITY_SCHEMA
        or raw["authority_version"] != 1
        or raw["artifact_scope"] != AUTHORITY_SCOPE
        or raw["scheme"] != AUTHORITY_SCHEME
    ):
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "authority schema or scope is invalid"
        )
    content = {key: raw[key] for key in raw if key != "authority_digest"}
    authority_digest = sha256_digest(raw["authority_digest"], "authority digest")
    if hashlib.sha256(canonical_authority_bytes(content)).hexdigest() != authority_digest:
        raise authority_violation(RolloutCode.R_HASH_MISMATCH, "authority digest drift")
    approved_by = sha256_digest(raw["approved_by"], "authority signer")
    trust_digest = sha256_digest(raw["trust_anchor_sha256"], "trust anchor digest")
    if approved_by != trust_digest:
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "authority signer identity drift"
        )
    signature = read_regular(signature_path, "detached authority signature")
    if not trust_store.verify(approved_by, AUTHORITY_SCHEME, encoded, signature.hex()):
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "authority signature is untrusted"
        )
    approved_at = parse_timestamp(raw["approved_at"], "approved_at")
    valid_from = parse_timestamp(raw["valid_from"], "valid_from")
    expires_at = parse_timestamp(raw["expires_at"], "expires_at")
    clock = datetime.now(timezone.utc) if now is None else now
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "verification clock lacks timezone"
        )
    if (
        approved_at > valid_from
        or clock < valid_from
        or clock >= expires_at
        or expires_at - valid_from != timedelta(hours=24)
    ):
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "authority validity is not current 24h"
        )
    runtime = parse_mapping(raw["runtime"], RUNTIME_FIELDS, "runtime")
    observed_runtime = observe_authority_runtime()
    if dict(runtime) != observed_runtime.as_document():
        raise authority_violation(
            RolloutCode.R_PROVIDER_MISMATCH,
            "signed process-isolated runtime identity drift",
        )
    runtime_policy = ReadOnlyRuntimePolicy(
        observed_runtime.feetech_servo_sdk_distribution,
        observed_runtime.feetech_servo_sdk_version,
        observed_runtime.pyserial_distribution,
        observed_runtime.pyserial_version,
        observed_runtime.scservo_sdk_distribution,
        observed_runtime.scservo_sdk_module,
        observed_runtime.scservo_sdk_origin,
        observed_runtime.scservo_sdk_origin_sha256,
    )
    profile = parse_mapping(
        raw["profile"], frozenset({"canonical_path", "content_sha256"}), "profile"
    )
    follower = parse_mapping(
        raw["follower"],
        frozenset(
            {
                "device_path",
                "device_identity_digest",
                "calibration_id",
                "calibration_path",
                "calibration_sha256",
            }
        ),
        "follower",
    )
    camera = parse_mapping(
        raw["camera"],
        frozenset({"device_path", "device_identity_digest", "width", "height", "fps"}),
        "camera",
    )
    thresholds = parse_mapping(
        raw["thresholds"],
        frozenset(
            {
                "camera_readiness_timeout_seconds",
                "joint_connect_timeout_seconds",
                "sample_pair_completion_timeout_seconds",
                "shutdown_grace_seconds",
                "camera_priming_frame_count",
                "accepted_sample_pair_count",
                "sample_max_age_seconds",
                "sample_max_skew_seconds",
                "max_fk_residual_m",
                "max_reprojection_error_px",
                "max_correspondence_error_px",
                "min_correspondences",
            }
        ),
        "thresholds",
    )
    priming_count = positive_integer(
        thresholds["camera_priming_frame_count"], "camera priming frame count"
    )
    accepted_pair_count = positive_integer(
        thresholds["accepted_sample_pair_count"], "accepted sample pair count"
    )
    if priming_count != 1 or accepted_pair_count != 2:
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED,
            "read-only capture cardinality must be exactly one priming frame and two pairs",
        )
    permissions = parse_mapping(
        raw["permissions"], frozenset({"camera", "follower", "forbidden"}), "permissions"
    )
    camera_permissions = tuple(cast("list[str]", permissions["camera"]))
    follower_permissions = tuple(cast("list[str]", permissions["follower"]))
    forbidden = tuple(cast("list[str]", permissions["forbidden"]))
    if (
        camera_permissions != CAMERA_PERMISSIONS
        or follower_permissions
        not in {FOLLOWER_PERMISSIONS, MANUAL_POSITIONING_FOLLOWER_PERMISSIONS}
        or forbidden != FORBIDDEN_CAPABILITIES
    ):
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "authority capabilities are invalid"
        )
    profile_path, calibration_path, follower_path, camera_path = verify_current_bindings(
        profile, follower, camera
    )
    return ProductionReadOnlyAcquisitionAuthority(
        READ_ONLY_CONSTRUCTION_SEAL,
        AUTHORITY_SCHEMA,
        required_text(raw["authority_id"], "authority_id"),
        AUTHORITY_SCOPE,
        approved_by,
        approved_at,
        valid_from,
        expires_at,
        sha256_digest(raw["source_lineage_authority_digest"], "source lineage authority"),
        sha256_digest(raw["provider_digest"], "provider digest"),
        runtime_policy,
        profile_path,
        sha256_digest(profile["content_sha256"], "profile digest"),
        follower_path,
        sha256_digest(follower["device_identity_digest"], "follower identity"),
        required_text(follower["calibration_id"], "calibration id"),
        calibration_path,
        sha256_digest(follower["calibration_sha256"], "calibration digest"),
        camera_path,
        sha256_digest(camera["device_identity_digest"], "camera identity"),
        positive_integer(camera["width"], "camera width"),
        positive_integer(camera["height"], "camera height"),
        positive_number(camera["fps"], "camera fps"),
        ReadOnlyTimingPolicy(
            CameraReadinessTimeoutSeconds(
                positive_number(thresholds["camera_readiness_timeout_seconds"], "camera readiness")
            ),
            JointConnectTimeoutSeconds(
                positive_number(thresholds["joint_connect_timeout_seconds"], "joint connect")
            ),
            PairCompletionTimeoutSeconds(
                positive_number(
                    thresholds["sample_pair_completion_timeout_seconds"],
                    "sample pair completion",
                )
            ),
            ShutdownGraceSeconds(
                positive_number(thresholds["shutdown_grace_seconds"], "shutdown grace")
            ),
            positive_number(thresholds["sample_max_age_seconds"], "sample max age"),
            positive_number(thresholds["sample_max_skew_seconds"], "sample max skew"),
        ),
        ReadOnlyCapturePolicy(priming_count, accepted_pair_count),
        ReadOnlyCameraPolicy(
            positive_number(thresholds["max_reprojection_error_px"], "reprojection error"),
            positive_integer(thresholds["min_correspondences"], "minimum correspondences"),
            positive_number(thresholds["max_correspondence_error_px"], "correspondence error"),
        ),
        ReadOnlyKinematicsPolicy(positive_number(thresholds["max_fk_residual_m"], "FK residual")),
        camera_permissions,
        follower_permissions,
        forbidden,
        authority_digest,
        trust_digest,
    )
