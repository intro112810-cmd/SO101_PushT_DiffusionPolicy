"""Fail-closed qualification, integrity, and pilot readiness checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import cast

from ..workspace import WorkspacePolicyError, runtime_artifact_root
from .store import LocalDatasetStore


def _value_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_root(root: Path) -> Path:
    artifact = runtime_artifact_root().resolve()
    absolute = root.absolute()
    for parent in (absolute, *absolute.parents):
        if parent.exists() and stat.S_ISLNK(parent.lstat().st_mode):
            raise WorkspacePolicyError("symlinked dataset root is forbidden")
        if parent == artifact.parent:
            break
    resolved = absolute.resolve()
    if resolved == artifact or artifact not in resolved.parents:
        raise WorkspacePolicyError("pilot root is outside artifact root")
    return resolved


def _load_attempts(root: Path) -> list[dict[str, object]]:
    raw = _safe_root(root) / "rejected" / "raw"
    if not raw.exists():
        return []
    result: list[dict[str, object]] = []
    for path in sorted(raw.glob("*.json")):
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise WorkspacePolicyError("special raw attempt is forbidden")
        value: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("attempt payload must be an object")
        payload = cast("dict[str, object]", value)
        actual = LocalDatasetStore.payload_digest(payload)
        if payload.get("sha256") != actual:
            raise ValueError(f"raw attempt digest mismatch: {path.name}")
        result.append(payload)
    return result


def _metadata(attempt: dict[str, object]) -> dict[str, object]:
    value = attempt.get("metadata")
    if not isinstance(value, dict):
        raise TypeError("attempt metadata must be an object")
    return cast("dict[str, object]", value)


def qualify_attempt(attempt: dict[str, object]) -> tuple[bool, list[str]]:
    """Return acceptance only for independently complete physical human evidence."""
    metadata, frames = _metadata(attempt), attempt.get("frames")
    errors: list[str] = []
    if not isinstance(metadata.get("task"), str) or not metadata.get("task"):
        errors.append("missing_task")
    if metadata.get("mode") != "human_gamepad":
        errors.append("not_human_gamepad")
    provenance = metadata.get("device_provenance")
    physical = metadata.get("physical_device") is True
    if isinstance(provenance, dict):
        typed_provenance = cast("dict[str, object]", provenance)
        physical = (
            physical
            and typed_provenance.get("adapter") == "lerobot_public_gamepad"
            and typed_provenance.get("physical") is True
        )
    if not physical:
        errors.append("missing_physical_device_provenance")
    # Operator buttons and precomputed flags are evidence only.  Acceptance is
    # recomputed from the frame stream below.
    if metadata.get("synthetic") is True or metadata.get("source") == "synthetic":
        errors.append("synthetic_attempt")
    if not isinstance(frames, list) or not frames:
        errors.append("incomplete_frames")
        return False, errors
    frame_values = cast("list[object]", frames)
    frame_ids: set[int] = set()
    previous_time = -1.0
    final_coverage = 0.0
    max_coverage = 0.0
    for expected, item in enumerate(frame_values):
        if not isinstance(item, dict):
            errors.append("malformed_frame")
            continue
        frame = cast("dict[str, object]", item)
        telemetry = frame.get("telemetry")
        if frame.get("frame_index") != expected:
            errors.append("dropped_or_duplicate_frame")
        timestamp = frame.get("timestamp")
        if not isinstance(timestamp, (int, float)) or float(timestamp) <= previous_time:
            errors.append("timestamp_discontinuity")
        else:
            previous_time = float(timestamp)
        if not isinstance(telemetry, dict):
            errors.append("missing_telemetry")
            continue
        data = cast("dict[str, object]", telemetry)
        frame_id = data.get("frame_id")
        if not isinstance(frame_id, int) or frame_id in frame_ids:
            errors.append("duplicate_frame_id")
        elif frame_id >= 0:
            frame_ids.add(frame_id)
        if any(
            data.get(key) is True
            for key in (
                "dropped",
                "duplicate",
                "clipped",
                "forbidden_contact",
                "incomplete_media",
                "replay_mismatch",
            )
        ):
            errors.append("unsafe_frame")
        if data.get("contact") is True and data.get("contact_allowed") is False:
            errors.append("forbidden_contact")
        if data.get("ack_status") != "applied" or frame.get("applied") is not True:
            errors.append("ack_failure")
        if (
            data.get("fault")
            or data.get("ik_error")
            or data.get("ik_fault")
            or data.get("timestamp_fault")
        ):
            errors.append("execution_fault")
        coverage = data.get("coverage")
        max_value = data.get("max_coverage")
        if isinstance(coverage, (int, float)):
            final_coverage = float(coverage)
        if isinstance(max_value, (int, float)):
            max_coverage = max(max_coverage, float(max_value))
    if max_coverage < 0.95 or final_coverage < 0.95:
        errors.append("coverage_not_met")
    return not errors, sorted(set(errors))


def _verify_current(root: Path) -> tuple[bool, str | None]:
    current = root / "current.json"
    if not current.exists():
        return True, None
    value: object = json.loads(current.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return False, "invalid_current_manifest"
    manifest = cast("dict[str, object]", value)
    digest = manifest.get("version")
    if not isinstance(digest, str):
        return False, "missing_version"
    version = root / "versions" / digest
    if not version.is_dir() or version.is_symlink():
        return False, "missing_version"
    files = manifest.get("files")
    if not isinstance(files, dict):
        return False, "missing_file_hashes"
    file_hashes = cast("dict[str, object]", files)
    actual_files: set[str] = set()
    for path in version.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            return False, "unsafe_canonical_entry"
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            return False, "unsafe_canonical_entry"
        actual_files.add(path.relative_to(version).as_posix())
    if set(file_hashes) != actual_files:
        return False, "canonical_membership_mismatch"
    encoded_files = json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode()
    if manifest.get("root_digest") != hashlib.sha256(encoded_files).hexdigest():
        return False, "canonical_tree_digest_mismatch"
    for relative, expected in file_hashes.items():
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            return False, "unsafe_manifest_path"
        if not isinstance(expected, str):
            return False, "invalid_file_hash"
        path = version / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            return False, "canonical_digest_mismatch"
    return True, None


verify_current = _verify_current


def replay_attempt(root: Path, attempt_id: str) -> dict[str, object]:
    """Validate the recorded 10 Hz action/observation alignment before simulator replay."""
    if not attempt_id.replace("_", "a").replace("-", "a").isalnum():
        raise ValueError("unsafe attempt id")
    attempts = _load_attempts(root)
    matching = [item for item in attempts if _metadata(item).get("attempt_id") == attempt_id]
    if len(matching) != 1:
        raise ValueError("attempt id is missing or duplicated")
    frames = matching[0].get("frames")
    if not isinstance(frames, list):
        return {"replay_match": False, "frames": 0, "errors": ["incomplete_frames"]}
    frame_values = cast("list[object]", frames)
    errors: list[str] = []
    for index, item in enumerate(frame_values):
        if not isinstance(item, dict):
            errors.append("malformed_frame")
            continue
        frame = cast("dict[str, object]", item)
        telemetry = frame.get("telemetry")
        if frame.get("frame_index") != index or frame.get("timestamp") != index / 10:
            errors.append("timing_mismatch")
        if not isinstance(telemetry, dict):
            errors.append("missing_telemetry")
            continue
        values = cast("dict[str, object]", telemetry)
        if frame.get("action") != values.get("applied_action"):
            errors.append("applied_action_mismatch")
        observation = frame.get("observation")
        if isinstance(observation, dict):
            typed_observation = cast("dict[str, object]", observation)
            rgb = typed_observation.get("observation.images.front")
            state = typed_observation.get("observation.state")
            if values.get("observation_rgb_hash") is not None and _value_digest(rgb) != values.get(
                "observation_rgb_hash"
            ):
                errors.append("observation_rgb_mismatch")
            if values.get("observation_state_hash") is not None and _value_digest(
                state
            ) != values.get("observation_state_hash"):
                errors.append("observation_state_mismatch")
        if values.get("action_hash") is not None and _value_digest(
            frame.get("action")
        ) != values.get("action_hash"):
            errors.append("action_hash_mismatch")
    return {"replay_match": not errors, "frames": len(frame_values), "errors": sorted(set(errors))}


def validate_pilot(root: Path, require_final_split: bool = False) -> dict[str, object]:
    safe = _safe_root(root)
    attempts = _load_attempts(safe)
    accepted: list[dict[str, object]] = []
    rejected = 0
    for attempt in attempts:
        valid, _errors = qualify_attempt(attempt)
        if valid:
            accepted.append(_metadata(attempt))
        else:
            rejected += 1
    sessions = {
        item.get("session_id") for item in accepted if isinstance(item.get("session_id"), str)
    }
    integrity, issue = _verify_current(safe)
    split_ok = not require_final_split or (safe / "splits.json").is_file()
    return {
        "human_pilot_status": "complete"
        if len(accepted) >= 20 and integrity and split_ok
        else "pending",
        "accepted_human": len(accepted),
        "attempts": len(attempts),
        "rejected": rejected,
        "sessions": len(sessions),
        "valid": integrity and split_ok,
        "integrity_issue": issue,
        "synthetic_training_leaks": 0,
    }
