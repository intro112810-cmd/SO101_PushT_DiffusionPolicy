"""Authentic two-step history assembly from validated provenance documents.

Owns the frame decode, affine agent-pose mapping, and assembly of exactly two
monotonic, provenance-bound observations.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from so101_pusht_benchmark.real_shadow import (
    physical_crop_to_checkpoint_image,
    validate_shadow_agent_pos,
)
from so101_pusht_benchmark.sim_to_real.joint_mapping import (
    JOINT_ORDER,
    affine_map_without_clipping,
)
from so101_pusht_benchmark.sim_to_real.replay_receipts import (
    parse_sample_document,
    require_digest,
    require_float,
    sha256_bytes,
    validate_camera_receipt,
    validate_joint_receipt,
    validate_lineage_receipt,
)
from so101_pusht_benchmark.sim_to_real.replay_types import (
    CAMERA_REGISTRATION_DIGEST,
    JOINT_EQUIVALENCE_DIGEST,
    Float32Vector,
    HistoryEvidence,
    HistoryStep,
    UInt8Image,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation


def _ordered_sources(evidence: HistoryEvidence) -> tuple[Path, Path]:
    """Resolve the two frames using the lineage-receipt identity as a gate."""
    artifact_id = evidence.lineage_document.get("artifact_id")
    if artifact_id not in {
        "fixture-local-dp_cnn-recovered-v3-seed0",
        "local-dp_cnn-recovered-v3-seed0",
        "local-dp_cnn-recovered-v4-seed0",
    }:
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "camera source lineage drift")
    first = evidence.source_frame_path
    second = first.parent / "secondary_physical_frame.jpg"
    if not first.is_file() or not second.is_file():
        raise RolloutViolation(RolloutCode.R_MISSING, "camera frames missing")
    return first, second


def _affine_agent_pos(body: Sequence[float]) -> Float32Vector:
    """Map body degrees through the receipt-bound unclipped affine formula."""
    ranges: dict[str, tuple[float, float]] = {
        "shoulder_pan": (-90.94505494505495, 90.94505494505495),
        "shoulder_lift": (-103.07692307692308, 103.07692307692308),
        "elbow_flex": (-105.27472527472527, 105.27472527472527),
        "wrist_flex": (-99.56043956043956, 99.56043956043956),
        "wrist_roll": (-166.8131868131868, 166.8131868131868),
    }
    q_ranges: dict[str, tuple[float, float]] = {
        "shoulder_pan": (-1.9198621771937616, 1.9198621771937634),
        "shoulder_lift": (-1.7453292519943224, 1.7453292519943366),
        "elbow_flex": (-1.69, 1.69),
        "wrist_flex": (-1.6580628494556928, 1.6580627293335335),
        "wrist_roll": (-2.7438472969992493, 2.841206309382605),
    }
    mapped: list[float] = []
    for joint, degree in zip(JOINT_ORDER, body, strict=True):
        degree_min, degree_max = ranges[joint]
        q_min, q_max = q_ranges[joint]
        mapped.append(
            affine_map_without_clipping(
                degree,
                degree_min=degree_min,
                degree_max=degree_max,
                q_min=q_min,
                q_max=q_max,
            )
        )
    return validate_shadow_agent_pos(np.asarray(mapped, dtype=np.float32))


def _camera_hashes(source: Path) -> tuple[str, str]:
    images = _decode_frames(source)
    first = physical_crop_to_checkpoint_image(images[0])
    second = physical_crop_to_checkpoint_image(images[1])
    return sha256_bytes(first.tobytes()), sha256_bytes(second.tobytes())


def _decode_frames(source: Path) -> tuple[UInt8Image, UInt8Image]:
    second_source = source.parent / "secondary_physical_frame.jpg"
    first = cv2.imread(str(source), cv2.IMREAD_COLOR)
    second = cv2.imread(str(second_source), cv2.IMREAD_COLOR)
    if first is None or second is None:
        raise RolloutViolation(RolloutCode.R_MISSING, "cannot read fixture camera frames")
    for label, image in (("first", first), ("second", second)):
        if image.shape != (400, 400, 3) or image.dtype != np.uint8:
            raise RolloutViolation(
                RolloutCode.HISTORY_INCOMPLETE, f"{label} frame must be uint8[400,400,3]"
            )
    return first, second


def build_history(evidence: HistoryEvidence) -> tuple[HistoryStep, HistoryStep]:
    """Assemble exactly two authentic, monotonic, provenance-bound observations."""
    lineage = validate_lineage_receipt(
        evidence.lineage_document,
        expected_digest=evidence.lineage_authority_digest,
    )
    fixture_only = bool(lineage["fixture_only"])
    validate_joint_receipt(
        evidence.joint_document,
        expected_digest=JOINT_EQUIVALENCE_DIGEST if fixture_only else None,
    )
    validate_camera_receipt(
        evidence.camera_document,
        expected_digest=CAMERA_REGISTRATION_DIGEST if fixture_only else None,
        expected_scope=(
            "synthetic_test_fixture"
            if fixture_only
            else "authorized_physical_diagnostic"
        ),
    )
    if len(evidence.samples) != 2:
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "exactly two samples required")
    records = [parse_sample_document(raw) for raw in evidence.samples]
    first_timestamp = require_float(records[0].get("camera_timestamp"), "camera_timestamp")
    second_timestamp = require_float(records[1].get("camera_timestamp"), "camera_timestamp")
    if not first_timestamp < second_timestamp:
        raise RolloutViolation(
            RolloutCode.HISTORY_INCOMPLETE, "sample timestamps must be monotonic"
        )
    digests = [require_digest(record["digest"], "sample digest") for record in records]
    frame_digests = [require_digest(record["frame_digest"], "frame_digest") for record in records]
    identity_set: set[tuple[str, tuple[float, ...]]] = set()
    for record in records:
        identity_set.add(
            (
                str(record["frame_digest"]),
                tuple(
                    require_float(item, "body_degrees")
                    for item in cast("list[object]", record["body_degrees"])
                ),
            )
        )
    if len(identity_set) != 2:
        raise RolloutViolation(RolloutCode.R_DUPLICATE_SAMPLE, "duplicate sample snapshot")
    if len(set(digests)) != 2:
        raise RolloutViolation(RolloutCode.R_DUPLICATE_HISTORY, "duplicate sample digest")
    if len(set(frame_digests)) != 2:
        raise RolloutViolation(RolloutCode.R_DUPLICATE_HISTORY, "duplicate camera frame")
    source, _ = _ordered_sources(evidence)
    camera_hashes = _camera_hashes(source)
    if lineage["fixture_only"]:
        for frame_digest, content_hash in zip(frame_digests, camera_hashes, strict=True):
            if frame_digest != content_hash:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "sample frame digest drift")
    decoded = _decode_frames(source)
    steps: list[HistoryStep] = []
    for index, (record, camera_sha256) in enumerate(zip(records, camera_hashes, strict=True)):
        checkpoint_image = physical_crop_to_checkpoint_image(decoded[index])
        agent_pos = _affine_agent_pos(
            tuple(
                require_float(value, "body_degrees")
                for value in cast("list[object]", record["body_degrees"])
            )
        )
        steps.append(
            HistoryStep(
                sample_id=str(record["record_id"]),
                sample_digest=require_digest(record["digest"], "sample digest"),
                frame_digest=require_digest(record["frame_digest"], "frame_digest"),
                camera_sha256=camera_sha256,
                agent_pos_sha256=sha256_bytes(agent_pos.tobytes()),
                checkpoint_image=checkpoint_image,
                agent_pos=agent_pos,
            )
        )
    return steps[0], steps[1]
