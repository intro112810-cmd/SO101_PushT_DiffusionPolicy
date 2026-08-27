"""Detached owner authority for the Torque_Enable manual-positioning read gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import cast

from .joint_corpus_contract import JOINT_CORPUS_PROVIDER_DIGEST, TORQUE_PERMISSION
from .policy_approval import ProductionTrustStore
from .read_only_authority import ProductionReadOnlyAcquisitionAuthority
from .read_only_authority_io import (
    parse_mapping,
    parse_timestamp,
    read_regular,
    required_text,
    sha256_digest,
)
from .read_only_authority_types import canonical_authority_bytes
from .rollout_codes import RolloutCode, RolloutViolation

SCHEMA = "so101-joint-corpus-manual-positioning-authority-v1"
SCOPE = "read_only_manual_positioning_gate"
SCHEME = "rsa-pkcs1v15-sha256-v1"
PERMISSIONS = (
    "direct_bus_connect",
    TORQUE_PERMISSION,
    "sync_read:Present_Position",
    "disconnect:disable_torque=false",
)
_FIELDS = frozenset(
    {
        "schema",
        "artifact_scope",
        "authority_id",
        "approved_by",
        "valid_from",
        "expires_at",
        "acquisition_authority_digest",
        "source_lineage_authority_digest",
        "provider_digest",
        "follower_device_digest",
        "calibration_digest",
        "permissions",
        "scheme",
        "authority_digest",
    }
)


@dataclass(frozen=True, slots=True)
class JointPositioningAuthority:
    authority_id: str
    approved_by: str
    valid_from: datetime
    expires_at: datetime
    acquisition_authority_digest: str
    source_lineage_authority_digest: str
    provider_digest: str
    follower_device_digest: str
    calibration_digest: str
    permissions: tuple[str, ...]
    canonical_digest: str


def authority_document(
    base: ProductionReadOnlyAcquisitionAuthority,
    *,
    authority_id: str,
    approved_by: str,
    valid_from: datetime,
) -> dict[str, object]:
    """Build public signing bytes bound to one already verified base authority."""
    document: dict[str, object] = {
        "schema": SCHEMA,
        "artifact_scope": SCOPE,
        "authority_id": authority_id,
        "approved_by": approved_by,
        "valid_from": valid_from.astimezone(timezone.utc).isoformat(),
        "expires_at": (valid_from + timedelta(hours=24)).astimezone(timezone.utc).isoformat(),
        "acquisition_authority_digest": base.canonical_digest,
        "source_lineage_authority_digest": base.source_lineage_authority_digest,
        "provider_digest": JOINT_CORPUS_PROVIDER_DIGEST,
        "follower_device_digest": base.follower_device_digest,
        "calibration_digest": base.calibration_digest,
        "permissions": list(PERMISSIONS),
        "scheme": SCHEME,
    }
    document["authority_digest"] = hashlib.sha256(canonical_authority_bytes(document)).hexdigest()
    return document


def load_joint_positioning_authority(
    path: Path,
    *,
    signature_path: Path,
    trust_store: ProductionTrustStore,
    base: ProductionReadOnlyAcquisitionAuthority,
    now: datetime | None = None,
) -> JointPositioningAuthority:
    """Authenticate current exact read permissions and base identity bindings."""
    encoded = read_regular(path, "manual positioning authority")
    try:
        raw_value: object = json.loads(encoded)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "manual positioning authority is invalid"
        ) from exc
    raw = parse_mapping(raw_value, _FIELDS, "manual positioning authority")
    if canonical_authority_bytes(raw) != encoded:
        raise RolloutViolation(
            RolloutCode.R_HASH_MISMATCH, "manual positioning authority is noncanonical"
        )
    if raw["schema"] != SCHEMA or raw["artifact_scope"] != SCOPE or raw["scheme"] != SCHEME:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "manual positioning authority scope is invalid"
        )
    content = {key: raw[key] for key in raw if key != "authority_digest"}
    authority_digest = sha256_digest(raw["authority_digest"], "positioning authority digest")
    if hashlib.sha256(canonical_authority_bytes(content)).hexdigest() != authority_digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "positioning authority digest drift")
    signer = sha256_digest(raw["approved_by"], "positioning authority signer")
    signature = read_regular(signature_path, "manual positioning signature")
    if not trust_store.verify(signer, SCHEME, encoded, signature.hex()):
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "manual positioning authority is untrusted"
        )
    valid_from = parse_timestamp(raw["valid_from"], "positioning valid_from")
    expires_at = parse_timestamp(raw["expires_at"], "positioning expires_at")
    observed = datetime.now(timezone.utc) if now is None else now
    if (
        observed.tzinfo is None
        or observed < valid_from
        or observed >= expires_at
        or expires_at - valid_from != timedelta(hours=24)
    ):
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "manual positioning authority is not current 24h"
        )
    permissions_raw = raw["permissions"]
    if (
        not isinstance(permissions_raw, list)
        or tuple(cast("list[object]", permissions_raw)) != PERMISSIONS
    ):
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "manual positioning permissions are invalid"
        )
    bindings = (
        (raw["acquisition_authority_digest"], base.canonical_digest),
        (raw["source_lineage_authority_digest"], base.source_lineage_authority_digest),
        (raw["provider_digest"], JOINT_CORPUS_PROVIDER_DIGEST),
        (raw["follower_device_digest"], base.follower_device_digest),
        (raw["calibration_digest"], base.calibration_digest),
    )
    if any(actual != expected for actual, expected in bindings):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "manual positioning identity drift")
    return JointPositioningAuthority(
        required_text(raw["authority_id"], "positioning authority id"),
        signer,
        valid_from,
        expires_at,
        base.canonical_digest,
        base.source_lineage_authority_digest,
        JOINT_CORPUS_PROVIDER_DIGEST,
        base.follower_device_digest,
        base.calibration_digest,
        PERMISSIONS,
        authority_digest,
    )
