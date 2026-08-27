"""Strict JSON primitives and current path bindings for read-only authority."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import stat
from typing import cast

from .read_only_authority_types import canonical_authority_bytes
from .rollout_codes import RolloutCode, RolloutViolation

__all__ = (
    "authority_violation",
    "parse_mapping",
    "parse_timestamp",
    "path_metadata_digest",
    "positive_integer",
    "positive_number",
    "read_regular",
    "required_text",
    "sha256_digest",
    "verify_current_bindings",
)


def authority_violation(code: RolloutCode, detail: str) -> RolloutViolation:
    return RolloutViolation(code, detail)


def parse_mapping(value: object, fields: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise authority_violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"{label} must be a mapping")
    result = cast("Mapping[str, object]", value)
    if frozenset(result) != fields:
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED,
            f"{label} fields are incomplete or unknown",
        )
    return result


def required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise authority_violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"{label} must be text")
    return value


def sha256_digest(value: object, label: str) -> str:
    result = required_text(value, label).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise authority_violation(RolloutCode.R_HASH_MISMATCH, f"{label} must be sha256")
    return result


def positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise authority_violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"{label} must be positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise authority_violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"{label} must be positive")
    return result


def positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, f"{label} must be positive integer"
        )
    return value


def parse_timestamp(value: object, label: str) -> datetime:
    raw = required_text(value, label)
    try:
        result = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise authority_violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"{label} is invalid") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise authority_violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"{label} lacks timezone")
    return result.astimezone(timezone.utc)


def absolute_path(value: object, label: str) -> Path:
    path = Path(required_text(value, label))
    if not path.is_absolute() or ".." in path.parts:
        raise authority_violation(RolloutCode.R_POLICY_UNAUTHORIZED, f"{label} is not canonical")
    return path


def path_metadata_digest(path: Path) -> str:
    """Derive a device identity from path metadata without opening the device."""
    lexical = path.absolute()
    try:
        lexical_info = lexical.lstat()
        link_target = str(lexical.readlink()) if stat.S_ISLNK(lexical_info.st_mode) else ""
        resolved = lexical.resolve(strict=True)
        target = resolved.stat()
    except OSError as exc:
        raise authority_violation(
            RolloutCode.R_MISSING, f"identity path is unavailable: {path}"
        ) from exc
    payload = {
        "lexical": str(lexical),
        "link_target": link_target,
        "resolved": str(resolved),
        "mode": target.st_mode,
        "device": target.st_dev,
        "inode": target.st_ino,
        "size": target.st_size,
        "rdev_major": os.major(target.st_rdev) if stat.S_ISCHR(target.st_mode) else None,
        "rdev_minor": os.minor(target.st_rdev) if stat.S_ISCHR(target.st_mode) else None,
    }
    return hashlib.sha256(canonical_authority_bytes(payload)).hexdigest()


def read_regular(path: Path, label: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError(label)
        return path.read_bytes()
    except OSError as exc:
        raise authority_violation(
            RolloutCode.R_POLICY_UNAUTHORIZED, f"cannot read {label}"
        ) from exc


def verify_current_bindings(
    profile: Mapping[str, object],
    follower: Mapping[str, object],
    camera: Mapping[str, object],
) -> tuple[Path, Path, Path, Path]:
    profile_path = absolute_path(profile["canonical_path"], "profile canonical_path")
    calibration_path = absolute_path(follower["calibration_path"], "calibration path")
    follower_path = absolute_path(follower["device_path"], "follower device path")
    camera_path = absolute_path(camera["device_path"], "camera device path")
    if hashlib.sha256(read_regular(profile_path, "profile")).hexdigest() != sha256_digest(
        profile["content_sha256"], "profile digest"
    ):
        raise authority_violation(RolloutCode.R_HASH_MISMATCH, "profile content drift")
    if hashlib.sha256(read_regular(calibration_path, "calibration")).hexdigest() != sha256_digest(
        follower["calibration_sha256"], "calibration digest"
    ):
        raise authority_violation(RolloutCode.R_HASH_MISMATCH, "calibration content drift")
    if path_metadata_digest(follower_path) != sha256_digest(
        follower["device_identity_digest"], "follower identity"
    ):
        raise authority_violation(RolloutCode.R_HASH_MISMATCH, "follower device identity drift")
    if path_metadata_digest(camera_path) != sha256_digest(
        camera["device_identity_digest"], "camera identity"
    ):
        raise authority_violation(RolloutCode.R_HASH_MISMATCH, "camera device identity drift")
    return profile_path, calibration_path, follower_path, camera_path
