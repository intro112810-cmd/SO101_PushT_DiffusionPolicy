"""Resumable, non-authoritative guided camera-registration capture."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .camera_registration import audit_camera_registration
from .camera_registration_target import (
    PLACEMENTS,
    Placement,
    local_inner_corners,
    placement_digest,
    table_corners,
    target_digest,
    target_document,
    validate_print_scale,
)
from .camera_registration_vision import (
    DetectedView,
    FittedGeometry,
    detect_checkerboard,
    encode_png,
    fit_geometry,
    reject_repeated_pose,
)
from .read_only_authority_types import ReadOnlyCameraPolicy
from .receipt_routing import (
    ReceiptRoutingError,
    locate_receipt_path,
    prepare_receipt_directory,
    validate_receipt_identity,
)
from .secure_io import atomic_write_new, unlink_owned_leaf
from .task_frame import invert_se2, registration_evidence_digest
from .rollout_codes import RolloutViolation

Image = NDArray[np.uint8]
TimestampClock = Callable[[], datetime]
DeviceProbe = Callable[[Path], str | None]
Prompt = Callable[[Placement], None]


@dataclass(frozen=True, slots=True)
class RegistrationAuthority:
    """Verified authority values consumed by the camera-only workflow."""

    authority_digest: str
    approved_by: str
    provider_digest: str
    profile_path: Path
    profile_digest: str
    camera_device_path: Path
    camera_device_digest: str
    calibration_digest: str
    width: int
    height: int
    fps: float
    crop: tuple[int, int, int, int]
    camera_policy: ReadOnlyCameraPolicy


@dataclass(frozen=True, slots=True)
class CameraObservation:
    width: int
    height: int
    fps: float


class RegistrationCamera(Protocol):
    def open(self) -> CameraObservation: ...

    def read(self) -> Image: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RegistrationCaptureRequest:
    root: Path
    measured_square_mm: float
    resume: bool
    production: bool = True


class Vision(Protocol):
    def detect(self, frame: Image, /) -> tuple[NDArray[np.float64], float]: ...

    def encode(self, frame: Image, /) -> bytes: ...

    def reject_repeat(
        self,
        candidate: NDArray[np.float64],
        accepted: list[DetectedView],
        resolution: tuple[int, int],
        /,
    ) -> None: ...

    def fit(self, views: list[DetectedView], resolution: tuple[int, int], /) -> FittedGeometry: ...


@dataclass(frozen=True, slots=True)
class OpenCvVision:
    detect: Callable[[Image], tuple[NDArray[np.float64], float]] = detect_checkerboard
    encode: Callable[[Image], bytes] = encode_png
    reject_repeat: Callable[[NDArray[np.float64], list[DetectedView], tuple[int, int]], None] = (
        reject_repeated_pose
    )
    fit: Callable[[list[DetectedView], tuple[int, int]], FittedGeometry] = fit_geometry


@dataclass(frozen=True, slots=True)
class CaptureDependencies:
    device_probe: DeviceProbe
    prompt: Prompt
    vision: Vision
    clock: TimestampClock


OPENCV_VISION = OpenCvVision()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def publish(path: Path, content: bytes) -> None:
    """Publish through the verified resolved target and retain lexical authority."""
    located = locate_receipt_path(path)
    location = validate_receipt_identity(located, production=located.canonical)
    published = atomic_write_new(
        location.resolved.parent,
        location.resolved.name,
        content,
        temporary=f".{location.resolved.name}.registration-{os.getpid()}.tmp",
    )
    try:
        current = locate_receipt_path(path)
        validate_receipt_identity(current, production=location.canonical)
    except (ReceiptRoutingError, RolloutViolation):
        unlink_owned_leaf(published)
        raise


def _prepare(root: Path, *, production: bool) -> Path:
    identity = validate_receipt_identity(locate_receipt_path(root), production=production)
    if production and not identity.lexical.is_relative_to(
        Path("/home/intro/InternLab/02_InTro_Project/04_experiments/")
        / "so101_pusht_benchmark/inference/sim_to_real_rollout/camera/raw"
    ):
        raise ValueError("registration sessions must be below canonical camera/raw")
    prepare_receipt_directory(identity.lexical, production=production)
    for name in ("members", "records"):
        prepare_receipt_directory(identity.lexical / name, production=production)
    return identity.lexical


def _record_paths(root: Path) -> list[Path]:
    records = root / "records"
    paths = sorted(records.glob("*.json"))
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("session record is not a regular file")
    return paths


def _load_records(root: Path, authority: RegistrationAuthority) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in _record_paths(root):
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("session record must be a mapping")
        records.append(cast("dict[str, object]", raw))
    if not records:
        return records
    header = records[0]
    if (
        header.get("record_type") != "session_header"
        or header.get("authority_digest") != authority.authority_digest
        or header.get("profile_digest") != authority.profile_digest
        or header.get("camera_device_digest") != authority.camera_device_digest
        or header.get("target_digest") != target_digest()
        or header.get("placement_digest") != placement_digest()
    ):
        raise ValueError("partial session identity drift")
    capture_ids: set[str] = set()
    for record in records[1:]:
        capture_id = record.get("capture_id")
        relative = record.get("path")
        declared = record.get("sha256")
        if (
            record.get("record_type") != "capture"
            or not isinstance(capture_id, str)
            or capture_id in capture_ids
            or not isinstance(relative, str)
            or not isinstance(declared, str)
        ):
            raise ValueError("partial session capture record is invalid")
        member = root / relative
        if member.is_symlink() or not member.is_file() or not member.is_relative_to(root):
            raise ValueError("partial session raw member is unsafe")
        if hashlib.sha256(member.read_bytes()).hexdigest() != declared:
            raise ValueError("partial session raw member tamper detected")
        capture_ids.add(capture_id)
    return records


def _header(authority: RegistrationAuthority, measured_square_mm: float) -> dict[str, object]:
    return {
        "record_type": "session_header",
        "schema": "so101-camera-registration-capture-session-v1",
        "authoritative": False,
        "publication_status": "partial_unsigned_non_authoritative",
        "authority_digest": authority.authority_digest,
        "approved_by": authority.approved_by,
        "provider_digest": authority.provider_digest,
        "profile_digest": authority.profile_digest,
        "camera_device_digest": authority.camera_device_digest,
        "calibration_digest": authority.calibration_digest,
        "target": target_document(),
        "target_digest": target_digest(),
        "placement_digest": placement_digest(),
        "operator_measured_square_mm": measured_square_mm,
    }


def _views(records: list[dict[str, object]]) -> list[DetectedView]:
    by_id = {placement.capture_id: placement for placement in PLACEMENTS}
    result: list[DetectedView] = []
    for record in records[1:]:
        capture_id = cast("str", record["capture_id"])
        raw_corners = cast("list[list[float]]", record["detected_corners_px"])
        result.append(DetectedView(by_id[capture_id], np.asarray(raw_corners, dtype=np.float64)))
    return result


def _capture_record(
    placement: Placement,
    corners: NDArray[np.float64],
    blur_variance: float,
    digest: str,
    timestamp: str,
) -> dict[str, object]:
    relative = f"members/{placement.capture_id}.png"
    return {
        "record_type": "capture",
        "capture_id": placement.capture_id,
        "phase": placement.phase,
        "role": placement.role,
        "path": relative,
        "sha256": digest,
        "timestamp": timestamp,
        "blur_variance": blur_variance,
        "detected_corners_px": corners.tolist(),
        "target_points_m": [list(point) for point in local_inner_corners()],
        "table_origin_xy_m": placement.table_origin_xy_m,
        "table_angle_degrees": placement.table_angle_degrees,
    }


def _corpus(
    records: list[dict[str, object]],
    authority: RegistrationAuthority,
    geometry: FittedGeometry,
) -> dict[str, object]:
    by_id = {placement.capture_id: placement for placement in PLACEMENTS}
    members: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    held_rows: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for record in records[1:]:
        placement = by_id[cast("str", record["capture_id"])]
        if placement.role is None:
            continue
        member = {key: record[key] for key in ("capture_id", "path", "sha256", "timestamp")}
        member["member_id"] = member.pop("capture_id")
        member["role"] = placement.role
        members.append(member)
        corners = cast("list[list[float]]", record["detected_corners_px"])
        table = table_corners(placement)
        target_rows = fit_rows if placement.role == "calibration_fit" else held_rows
        for index, (image_point, table_point) in enumerate(zip(corners, table, strict=True)):
            target_rows.append(
                {
                    "correspondence_id": f"{placement.capture_id}-{index:02d}",
                    "member_id": placement.capture_id,
                    "timestamp": record["timestamp"],
                    "image_point_px": image_point,
                    "table_point_m": list(table_point),
                    "simulation_point_m": [table_point[0], table_point[1]],
                }
            )
        if placement.role == "checkpoint_held_out":
            checkpoints.append(
                {
                    "member_id": placement.capture_id,
                    "sha256": record["sha256"],
                    "timestamp": record["timestamp"],
                }
            )
    se2 = geometry.physical_to_sim
    corpus: dict[str, object] = {
        "schema": "so101-camera-registration-corpus-v2",
        "evidence_scope": "production",
        "intrinsics": {
            "model": "pinhole_brown_conrady",
            "matrix": list(geometry.intrinsics),
            "units": "pixels",
        },
        "distortion": {
            "model": "brown_conrady",
            "coefficients": list(geometry.distortion),
            "order": ["k1", "k2", "p1", "p2", "k3"],
        },
        "camera_to_table": {
            "direction": "camera_to_table",
            "matrix": list(geometry.camera_to_table),
            "translation_units": "meters",
            "camera_axes": "x_right_y_down_z_forward",
            "table_axes": "x_right_y_forward_z_up",
        },
        "physical_to_sim": {
            "direction": "physical_table_to_simulation_table",
            "matrix_2x3": list(se2),
            "physical_units": "meters",
            "simulation_units": "meters",
        },
        "physical_to_sim_se2": list(se2),
        "sim_to_physical_se2": list(invert_se2(se2)),
        "members": members,
        "fit_correspondences": fit_rows,
        "held_out_correspondences": held_rows,
        "checkpoint_view_members": checkpoints,
        "device_hash": authority.camera_device_digest,
        "resolution": [authority.width, authority.height],
        "crop": list(authority.crop),
        "orientation_hash": placement_digest(),
        "config_hash": authority.profile_digest,
    }
    corpus["camera_digest"] = registration_evidence_digest(corpus)
    return corpus


def _finalize(
    root: Path,
    records: list[dict[str, object]],
    authority: RegistrationAuthority,
    vision: Vision,
) -> dict[str, object]:
    views = _views(records)
    geometry = vision.fit(views, (authority.width, authority.height))
    corpus = _corpus(records, authority, geometry)
    receipt = audit_camera_registration(
        corpus,
        corpus_root=root,
        source_scope="production",
        thresholds=authority.camera_policy,
    )
    publish(root / "corpus.json", _json_bytes(corpus))
    signing_request = {
        "schema": "so101-camera-registration-signing-request-v1",
        "authoritative": False,
        "publication_status": "capture_complete_owner_signature_required",
        "corpus_digest": corpus["camera_digest"],
        "acquisition_authority_digest": authority.authority_digest,
        "approved_by": authority.approved_by,
        "provider_digest": authority.provider_digest,
        "profile_digest": authority.profile_digest,
        "camera_device_digest": authority.camera_device_digest,
        "calibration_digest": authority.calibration_digest,
        "orientation_digest": placement_digest(),
        "required_next_artifacts": [
            "owner-signed production live identity",
            "owner-signed corpus-authority.json",
            "governed production audit receipt",
        ],
    }
    publish(root / "signing-request.json", _json_bytes(signing_request))
    summary = {
        **receipt,
        "schema": "so101-camera-registration-capture-summary-v1",
        "authoritative": False,
        "publication_status": "capture_complete_owner_signature_required",
        "raw_capture_count": len(views),
        "corpus_path": str(root / "corpus.json"),
        "signing_request_path": str(root / "signing-request.json"),
    }
    publish(root / "capture-summary.json", _json_bytes(summary))
    return summary


def registration_json_bytes(value: object) -> bytes:
    """Encode deterministic registration JSON for offline assembly."""
    return _json_bytes(value)


def prepare_registration_root(root: Path, *, production: bool) -> Path:
    """Prepare one registration root through the governed route."""
    return _prepare(root, production=production)


def registration_session_header(
    authority: RegistrationAuthority, measured_square_mm: float
) -> dict[str, object]:
    """Build the same signed-identity-bound header used by live capture."""
    return _header(authority, measured_square_mm)


def registration_capture_record(
    placement: Placement,
    corners: NDArray[np.float64],
    blur_variance: float,
    digest: str,
    timestamp: str,
) -> dict[str, object]:
    """Build one table/checkpoint record using the live capture contract."""
    return _capture_record(placement, corners, blur_variance, digest, timestamp)


def build_registration_corpus(
    records: list[dict[str, object]],
    authority: RegistrationAuthority,
    geometry: FittedGeometry,
) -> dict[str, object]:
    """Build the canonical corpus from governed records and fitted geometry."""
    return _corpus(records, authority, geometry)


def run_guided_capture(
    request: RegistrationCaptureRequest,
    authority: RegistrationAuthority,
    camera: RegistrationCamera,
    dependencies: CaptureDependencies,
) -> dict[str, object]:
    """Capture one event-driven session; incomplete records remain resumable and unsigned."""
    validate_print_scale(request.measured_square_mm)
    root = _prepare(request.root, production=request.production)
    records = _load_records(root, authority)
    if records and not request.resume:
        raise ValueError("session already exists; use --resume")
    if not records:
        publish(
            root / "records/000-session-header.json",
            _json_bytes(_header(authority, request.measured_square_mm)),
        )
        records = _load_records(root, authority)
    elif records[0].get("operator_measured_square_mm") != request.measured_square_mm:
        raise ValueError("resume print-scale measurement drift")
    if (root / "capture-summary.json").is_file():
        summary: object = json.loads((root / "capture-summary.json").read_text(encoding="utf-8"))
        return cast("dict[str, object]", summary)
    accepted = _views(records)
    completed = {view.placement.capture_id for view in accepted}
    observation = camera.open()
    try:
        if (observation.width, observation.height, observation.fps) != (
            authority.width,
            authority.height,
            authority.fps,
        ):
            raise ValueError("observed camera profile drift")
        camera.read()  # Exactly one discarded priming frame for this camera open.
        for placement in PLACEMENTS:
            if placement.capture_id in completed:
                continue
            observed_device = dependencies.device_probe(authority.camera_device_path)
            if observed_device != authority.camera_device_digest:
                raise ValueError("camera device identity drift")
            dependencies.prompt(placement)
            frame = camera.read()
            corners, blur_variance = dependencies.vision.detect(frame)
            dependencies.vision.reject_repeat(
                corners, accepted, (authority.width, authority.height)
            )
            png = dependencies.vision.encode(frame)
            digest = hashlib.sha256(png).hexdigest()
            publish(root / f"members/{placement.capture_id}.png", png)
            timestamp = (
                dependencies.clock()
                .astimezone(timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            )
            record = _capture_record(
                placement,
                corners,
                blur_variance,
                digest,
                timestamp,
            )
            index = len(records)
            publish(root / f"records/{index:03d}-{placement.capture_id}.json", _json_bytes(record))
            accepted.append(DetectedView(placement, corners))
            records.append(record)
    finally:
        camera.close()
    return _finalize(root, records, authority, dependencies.vision)
