"""Strict raw-member and correspondence parsing for camera evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import struct
from typing import cast

from .camera_geometry import CameraModel, invalid, mapping, parse_camera_model, sequence
from .rollout_codes import RolloutViolation
from .task_frame import parse_se2_material, registration_evidence_digest

_SCHEMA = "so101-camera-registration-corpus-v2"
_SCOPES = {"synthetic_test_fixture", "production"}
_TOP_FIELDS = {
    "schema",
    "evidence_scope",
    "intrinsics",
    "distortion",
    "camera_to_table",
    "physical_to_sim",
    "physical_to_sim_se2",
    "sim_to_physical_se2",
    "members",
    "fit_correspondences",
    "held_out_correspondences",
    "checkpoint_view_members",
    "device_hash",
    "resolution",
    "crop",
    "orientation_hash",
    "config_hash",
    "camera_digest",
}
_MEMBER_FIELDS = {"member_id", "path", "sha256", "timestamp", "role"}
_CORRESPONDENCE_FIELDS = {
    "correspondence_id",
    "member_id",
    "timestamp",
    "image_point_px",
    "table_point_m",
    "simulation_point_m",
}
_CHECKPOINT_FIELDS = {"member_id", "sha256", "timestamp"}


@dataclass(frozen=True, slots=True)
class ParsedCameraCorpus:
    digest: str
    resolution: tuple[int, int]
    members: dict[str, dict[str, object]]
    fit: list[dict[str, object]]
    held_out: list[dict[str, object]]
    model: CameraModel


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise invalid(f"{label} must be a nonempty string")
    return value


def digest(value: object, label: str) -> str:
    result = _text(value, label).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise invalid(f"{label} must be SHA-256")
    return result


def _timestamp(value: object, label: str) -> str:
    result = _text(value, label)
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as exc:
        raise invalid(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise invalid(f"{label} must include a timezone")
    return result


def _resolution(corpus: Mapping[str, object]) -> tuple[int, int]:
    values = sequence(corpus.get("resolution"), "resolution", 2)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values):
        raise invalid("resolution must contain positive integer dimensions")
    width, height = cast("list[int]", values)
    crop = sequence(corpus.get("crop"), "crop", 4)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in crop):
        raise invalid("crop must contain integer x,y,width,height")
    x, y, crop_width, crop_height = cast("list[int]", crop)
    if (
        x < 0
        or y < 0
        or crop_width <= 0
        or crop_height <= 0
        or x + crop_width > width
        or y + crop_height > height
    ):
        raise invalid("crop is outside the raw image resolution")
    return width, height


def _member_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise invalid("member path must be relative to the corpus")
    corpus_root = root.resolve()
    unresolved = corpus_root / relative
    candidate = unresolved.resolve()
    if (
        not candidate.is_relative_to(corpus_root)
        or unresolved.is_symlink()
        or not candidate.is_file()
    ):
        raise invalid("member path escapes the corpus or is not a regular file")
    return candidate


def _png_resolution(content: bytes, label: str) -> tuple[int, int]:
    if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n" or content[12:16] != b"IHDR":
        raise invalid(f"{label} is not a raw PNG image member")
    return struct.unpack(">II", content[16:24])


def _members(
    corpus: Mapping[str, object], root: Path, resolution: tuple[int, int]
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    paths: set[str] = set()
    for item in sequence(corpus.get("members"), "members"):
        member = mapping(item, "member")
        if set(member) != _MEMBER_FIELDS:
            raise invalid("member fields are incomplete or unknown")
        member_id = _text(member["member_id"], "member_id")
        relative = _text(member["path"], "member.path")
        role = _text(member["role"], "member.role")
        if role not in {"calibration_fit", "checkpoint_held_out"}:
            raise invalid("member role is invalid")
        if member_id in result or relative in paths:
            raise invalid("member identities and paths must be unique")
        content = _member_path(root, relative).read_bytes()
        declared = digest(member["sha256"], "member.sha256")
        if hashlib.sha256(content).hexdigest() != declared:
            raise invalid("raw image member digest mismatch")
        if _png_resolution(content, member_id) != resolution:
            raise invalid("raw image member resolution mismatch")
        result[member_id] = {
            "sha256": declared,
            "timestamp": _timestamp(member["timestamp"], "member.timestamp"),
            "role": role,
        }
        paths.add(relative)
    roles = [str(member["role"]) for member in result.values()]
    if roles.count("calibration_fit") < 2 or roles.count("checkpoint_held_out") < 2:
        raise invalid("fit and checkpoint-view evidence require at least two raw members each")
    return result


def _correspondences(
    corpus: Mapping[str, object],
    field: str,
    members: Mapping[str, Mapping[str, object]],
    role: str,
    minimum: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    identities: set[str] = set()
    covered: set[str] = set()
    for item in sequence(corpus.get(field), field):
        row = mapping(item, field)
        if set(row) != _CORRESPONDENCE_FIELDS:
            raise invalid(f"{field} fields are incomplete or unknown")
        identity = _text(row["correspondence_id"], f"{field}.correspondence_id")
        member_id = _text(row["member_id"], f"{field}.member_id")
        if identity in identities or member_id not in members:
            raise invalid(f"{field} identity or member reference is invalid")
        member = members[member_id]
        if (
            member["role"] != role
            or _timestamp(row["timestamp"], f"{field}.timestamp") != member["timestamp"]
        ):
            raise invalid(f"{field} member role or timestamp mismatch")
        identities.add(identity)
        covered.add(member_id)
        rows.append(row)
    eligible = {key for key, member in members.items() if member["role"] == role}
    if len(rows) < minimum or covered != eligible:
        raise invalid(f"{field} lacks policy count or raw-member coverage")
    return rows


def _checkpoint_members(
    corpus: Mapping[str, object], members: Mapping[str, Mapping[str, object]]
) -> None:
    observed: set[str] = set()
    for item in sequence(corpus.get("checkpoint_view_members"), "checkpoint_view_members"):
        checkpoint = mapping(item, "checkpoint_view_member")
        if set(checkpoint) != _CHECKPOINT_FIELDS:
            raise invalid("checkpoint-view member fields are incomplete or unknown")
        member_id = _text(checkpoint["member_id"], "checkpoint member_id")
        if member_id in observed or member_id not in members:
            raise invalid("checkpoint-view member identity is invalid")
        member = members[member_id]
        if (
            member["role"] != "checkpoint_held_out"
            or digest(checkpoint["sha256"], "checkpoint.sha256") != member["sha256"]
            or _timestamp(checkpoint["timestamp"], "checkpoint.timestamp") != member["timestamp"]
        ):
            raise invalid("checkpoint-view member digest or timestamp mismatch")
        observed.add(member_id)
    expected = {key for key, member in members.items() if member["role"] == "checkpoint_held_out"}
    if observed != expected:
        raise invalid("checkpoint-view member coverage is incomplete")


def parse_camera_corpus(
    corpus: Mapping[str, object],
    root: Path,
    minimum: int,
    *,
    expected_scope: str,
) -> ParsedCameraCorpus:
    if set(corpus) != _TOP_FIELDS or corpus.get("schema") != _SCHEMA:
        raise invalid("raw camera corpus schema fields are incomplete or unknown")
    if expected_scope not in _SCOPES or corpus.get("evidence_scope") != expected_scope:
        raise invalid("camera corpus scope does not match parser authority")
    resolution = _resolution(corpus)
    for label in ("device_hash", "orientation_hash", "config_hash"):
        digest(corpus.get(label), label)
    declared = digest(corpus.get("camera_digest"), "camera_digest")
    if registration_evidence_digest(corpus) != declared:
        raise invalid("camera registration corpus hash drift")
    try:
        material = parse_se2_material(corpus)
    except RolloutViolation as exc:
        raise invalid(f"camera table transform is invalid: {exc}") from exc
    if material.camera_digest != declared:
        raise invalid("camera transform is not bound to the corpus digest")
    members = _members(corpus, root, resolution)
    fit = _correspondences(corpus, "fit_correspondences", members, "calibration_fit", minimum)
    held = _correspondences(
        corpus, "held_out_correspondences", members, "checkpoint_held_out", minimum
    )
    _checkpoint_members(corpus, members)
    return ParsedCameraCorpus(
        declared, resolution, members, fit, held, parse_camera_model(corpus, resolution)
    )
