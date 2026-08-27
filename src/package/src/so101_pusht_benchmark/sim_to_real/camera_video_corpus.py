"""Build a camera-registration candidate from immutable recorded videos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .camera_registration import audit_camera_registration
from .camera_corpus import parse_camera_corpus
from .camera_geometry import evaluate_correspondences
from .rollout_codes import RolloutViolation
from .camera_registration_capture import (
    RegistrationAuthority,
    build_registration_corpus,
    registration_capture_record,
    registration_json_bytes,
    registration_session_header,
    prepare_registration_root,
    publish,
)
from .camera_registration_target import PLACEMENTS, Placement
from .camera_registration_vision import DetectedView, detect_checkerboard, encode_png, fit_geometry
from .intrinsic_extraction import scan_frames
from .intrinsic_extraction_io import decode_video, selected_images

Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class RecordedTableClip:
    """One immutable clip assigned to one exact placement contract."""

    placement_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class VideoCorpusRequest:
    """Inputs for deterministic offline camera corpus assembly."""

    intrinsic_evidence: Path
    table_clips: tuple[RecordedTableClip, ...]
    output_root: Path
    measured_square_mm: float


def _intrinsic_views(root: Path) -> list[DetectedView]:
    raw: object = json.loads((root / "selected-frames.json").read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("selected intrinsic evidence must be a list")
    views: list[DetectedView] = []
    for item in cast("list[object]", raw):
        if not isinstance(item, Mapping):
            continue
        typed = cast("Mapping[str, object]", item)
        if typed.get("role") != "fit":
            continue
        corners = np.asarray(typed.get("corners_px"), dtype=np.float64)
        if corners.shape != (35, 2) or not np.isfinite(corners).all():
            raise ValueError("selected intrinsic corners are invalid")
        rank = len(views) + 1
        placement = Placement(f"offline-intrinsic-{rank:02d}", "intrinsics", None, "offline")
        views.append(DetectedView(placement, corners))
    if len(views) < 6:
        raise ValueError("offline intrinsic evidence requires at least six fit views")
    return views


def _best_clip_view(path: Path) -> tuple[Image, NDArray[np.float64], float, float, int]:
    scan = scan_frames(decode_video(path), detect_checkerboard)
    if not scan.candidates:
        raise ValueError(f"recorded clip has no complete 35-corner frame: {path}")
    candidate = max(scan.candidates, key=lambda item: (item.sharpness, -item.frame_index))
    image = selected_images(path, (candidate.frame_index,))[0]
    return (
        image,
        candidate.corners,
        candidate.sharpness,
        candidate.timestamp_seconds,
        candidate.frame_index,
    )


def _timestamp(path: Path, offset_seconds: float) -> str:
    captured = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) + timedelta(
        seconds=offset_seconds
    )
    return captured.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diagnostic_metrics(
    corpus: dict[str, object], root: Path, authority: RegistrationAuthority
) -> dict[str, float]:
    parsed = parse_camera_corpus(
        corpus,
        root,
        authority.camera_policy.min_correspondences,
        expected_scope="production",
    )
    fit = evaluate_correspondences(
        parsed.model, parsed.fit, label="fit", resolution=parsed.resolution
    )
    held = evaluate_correspondences(
        parsed.model, parsed.held_out, label="held_out", resolution=parsed.resolution
    )
    return {
        "fit_reprojection_rmse_px": fit[0],
        "fit_reprojection_max_px": fit[1],
        "fit_physical_to_sim_rmse_m": fit[2],
        "fit_physical_to_sim_max_m": fit[3],
        "heldout_reprojection_rmse_px": held[0],
        "heldout_reprojection_max_px": held[1],
        "heldout_physical_to_sim_rmse_m": held[2],
        "heldout_physical_to_sim_max_m": held[3],
    }


def build_recorded_camera_corpus(
    request: VideoCorpusRequest,
    authority: RegistrationAuthority,
) -> dict[str, object]:
    """Select recorded frames, recompute geometry, and publish an unsigned candidate."""
    if request.output_root.exists() and any(request.output_root.iterdir()):
        raise ValueError("offline camera corpus output must be fresh")
    root = prepare_registration_root(request.output_root, production=True)
    placements = {placement.capture_id: placement for placement in PLACEMENTS}
    expected = {
        "table-fit-a",
        "table-fit-b",
        "table-fit-c",
        "checkpoint-held-a",
        "checkpoint-held-b",
    }
    if {clip.placement_id for clip in request.table_clips} != expected:
        raise ValueError("recorded table clips must cover the exact five placement ids")

    records: list[dict[str, object]] = [
        registration_session_header(authority, request.measured_square_mm)
    ]
    views = _intrinsic_views(request.intrinsic_evidence)
    sources: list[dict[str, object]] = []
    for clip in request.table_clips:
        placement = placements[clip.placement_id]
        image, corners, sharpness, offset, frame_index = _best_clip_view(clip.path)
        png = encode_png(image)
        digest = hashlib.sha256(png).hexdigest()
        publish(root / f"members/{placement.capture_id}.png", png)
        timestamp = _timestamp(clip.path, offset)
        record = registration_capture_record(placement, corners, sharpness, digest, timestamp)
        records.append(record)
        views.append(DetectedView(placement, corners))
        sources.append(
            {
                "placement_id": placement.capture_id,
                "source_video": str(clip.path.resolve()),
                "source_sha256": _source_digest(clip.path),
                "selected_frame_index": frame_index,
                "selected_frame_timestamp_seconds": offset,
                "selected_frame_sharpness": sharpness,
            }
        )

    geometry = fit_geometry(views, (authority.width, authority.height))
    corpus = build_registration_corpus(records, authority, geometry)
    diagnostics = _diagnostic_metrics(corpus, root, authority)
    for index, record in enumerate(records):
        name = (
            "000-session-header.json" if index == 0 else f"{index:03d}-{record['capture_id']}.json"
        )
        publish(root / f"records/{name}", registration_json_bytes(record))
    publish(root / "corpus.json", registration_json_bytes(corpus))
    intrinsic_count = sum(view.placement.phase == "intrinsics" for view in views)
    try:
        receipt = audit_camera_registration(
            corpus,
            corpus_root=root,
            source_scope="production",
            thresholds=authority.camera_policy,
        )
    except RolloutViolation as exc:
        failure = {
            "schema": "so101-camera-registration-recorded-video-summary-v1",
            "authoritative": False,
            "audited": False,
            "publication_status": "recorded_candidate_audit_failed",
            "audit_error": str(exc),
            "diagnostics": diagnostics,
            "intrinsic_evidence": str(request.intrinsic_evidence.resolve()),
            "intrinsic_fit_view_count": intrinsic_count,
            "recorded_sources": sources,
            "corpus_path": str((root / "corpus.json").resolve()),
        }
        publish(root / "capture-summary.json", registration_json_bytes(failure))
        raise
    summary = {
        **receipt,
        "diagnostics": diagnostics,
        "schema": "so101-camera-registration-recorded-video-summary-v1",
        "authoritative": False,
        "publication_status": "recorded_candidate_owner_signature_required",
        "intrinsic_evidence": str(request.intrinsic_evidence.resolve()),
        "intrinsic_fit_view_count": intrinsic_count,
        "recorded_sources": sources,
        "corpus_path": str((root / "corpus.json").resolve()),
    }
    publish(root / "capture-summary.json", registration_json_bytes(summary))
    return summary
