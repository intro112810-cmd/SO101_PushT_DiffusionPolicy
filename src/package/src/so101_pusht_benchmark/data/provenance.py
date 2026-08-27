"""Canonical provenance construction without recording telemetry as model features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


POLICY_FEATURES = frozenset({"observation.images.front", "observation.state", "action"})


@dataclass(frozen=True, slots=True)
class ProvenanceRequest:
    dataset_id: str
    source_digest: str
    config_digest: str
    scene_digest: str
    calibration_digest: str
    runtime: Mapping[str, str]
    features: Mapping[str, object]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_provenance(request: ProvenanceRequest) -> dict[str, object]:
    features = request.features
    if set(features) != set(POLICY_FEATURES):
        raise ValueError("canonical policy features must be image, state, action only")
    if any("telemetry" in key or "raw_" in key for key in features):
        raise ValueError("telemetry must remain outside the canonical normalizer")
    value: dict[str, object] = {
        "dataset_id": request.dataset_id,
        "source_digest": request.source_digest,
        "config_digest": request.config_digest,
        "scene_digest": request.scene_digest,
        "calibration_digest": request.calibration_digest,
        "runtime": dict(request.runtime),
        "features": dict(features),
        "lerobot_version": "0.4.4",
    }
    value["digest"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value
