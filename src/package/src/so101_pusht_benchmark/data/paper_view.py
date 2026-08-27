"""Deterministic atomic storage for the native pushT-so100 policy arrays."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..workspace import WorkspacePolicyError, runtime_artifact_root

EXPORTER_REVISION = "pusht_so100_native_v1"
FPS = 10
CHUNK_ROWS = {
    "cam_top": 1,
    "cam_side": 1,
    "agent_pos": 256,
    "action": 256,
    "timestamp": 1024,
    "episode_id": 1024,
    "frame_index": 1024,
    "episode_ends": 1024,
}
_SINGLE_CAM = os.environ.get("PUSHT_SINGLE_CAM") == "1"

if _SINGLE_CAM:
    ARRAY_NAMES = (
        "cam_top",
        "agent_pos",
        "action",
        "timestamp",
        "episode_id",
        "frame_index",
    )
else:
    ARRAY_NAMES = (
        "cam_top",
        "cam_side",
        "agent_pos",
        "action",
        "timestamp",
        "episode_id",
        "frame_index",
    )


class PaperViewError(ValueError):
    """Raised when native persisted data is unsafe, malformed, or ineligible."""


@dataclass(frozen=True, slots=True)
class PaperArray:
    values: NDArray[np.generic]
    unit: str


@dataclass(frozen=True, slots=True)
class PaperViewMetadata:
    canonical_digest: str
    root_digest: str
    root_provenance: dict[str, object]
    episode_ids: list[str]
    splits: dict[str, object]
    runtime_lock_digest: str
    training_eligible: bool


@dataclass(frozen=True, slots=True)
class LoadedPaperView:
    arrays: dict[str, NDArray[np.generic]]
    episode_ends: NDArray[np.int64]
    manifest: dict[str, object]
    splits: dict[str, object]


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def safe_paper_path(path: Path) -> Path:
    artifact = runtime_artifact_root().resolve()
    absolute = path.absolute()
    if ".." in path.parts:
        raise WorkspacePolicyError(f"parent traversal is forbidden: {path}")
    for parent in (absolute, *absolute.parents):
        if parent.exists() and stat.S_ISLNK(parent.lstat().st_mode):
            raise WorkspacePolicyError(f"symlink path is forbidden: {path}")
        if parent == artifact.parent:
            break
    resolved = absolute.resolve()
    if resolved == artifact or artifact not in resolved.parents:
        raise WorkspacePolicyError(f"native view must be beneath artifact root: {path}")
    return resolved


def require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PaperViewError(f"{label} must be lowercase SHA-256")
    return value


def trusted_runtime_lock_digest() -> str:
    try:
        from ..native_runtime import trusted_native_runtime_lock_digest

        return trusted_native_runtime_lock_digest()
    except Exception as exc:
        raise PaperViewError(f"trusted runtime lock provenance failed: {exc}") from exc


def _validated_root_provenance(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PaperViewError("root provenance must be an object")
    provenance = cast("dict[str, object]", value)
    if (
        set(provenance) != {"schema", "source_members", "episodes"}
        or provenance.get("schema") != "pusht-so100-root-provenance-v1"
    ):
        raise PaperViewError("root provenance fields are not exact")
    members_raw, episodes_raw = provenance.get("source_members"), provenance.get("episodes")
    if not isinstance(members_raw, dict) or not members_raw or not isinstance(episodes_raw, list):
        raise PaperViewError("root provenance inputs are missing")
    members = cast("dict[object, object]", members_raw)
    if not all(
        isinstance(path, str)
        and bool(path)
        and not Path(path).is_absolute()
        and ".." not in Path(path).parts
        and require_sha256(digest, f"root provenance member {path}") == digest
        for path, digest in members.items()
    ):
        raise PaperViewError("root provenance source members are malformed")
    if not episodes_raw:
        raise PaperViewError("root provenance episode inputs are missing")
    for raw in cast("list[object]", episodes_raw):
        if not isinstance(raw, dict):
            raise PaperViewError("root provenance episode is malformed")
        episode = cast("dict[str, object]", raw)
        if (
            set(episode) != {"episode_id", "length"}
            or not isinstance(episode.get("episode_id"), str)
            or not episode.get("episode_id")
            or not isinstance(episode.get("length"), int)
            or isinstance(episode.get("length"), bool)
            or cast(int, episode["length"]) <= 0
        ):
            raise PaperViewError("root provenance episode fields are malformed")
    return provenance


def root_provenance_digest(value: object) -> str:
    return hashlib.sha256(_json(_validated_root_provenance(value))).hexdigest()


def validate_root_provenance(
    value: object, episode_ids: list[str], episode_ends: NDArray[np.int64]
) -> dict[str, object]:
    provenance = _validated_root_provenance(value)
    episodes = cast("list[dict[str, object]]", provenance["episodes"])
    starts = [0, *episode_ends.tolist()[:-1]]
    expected = [
        {"episode_id": episode_id, "length": end - start}
        for episode_id, start, end in zip(episode_ids, starts, episode_ends.tolist(), strict=True)
    ]
    if episodes != expected:
        raise PaperViewError("root provenance episode descriptors mismatch")
    return provenance


def dtype_contract(array: NDArray[np.generic]) -> tuple[str, str]:
    if array.dtype == np.dtype(np.uint8):
        return "|u1", "uint8"
    if array.dtype == np.dtype(np.float32):
        return "<f4", "float32"
    if array.dtype == np.dtype(np.float64):
        return "<f8", "float64"
    if array.dtype == np.dtype(np.int64):
        return "<i8", "int64"
    raise PaperViewError("unsupported native array dtype")


def _array_metadata(array: NDArray[np.generic], chunks: tuple[int, ...]) -> dict[str, object]:
    return {
        "chunks": list(chunks),
        "compressor": None,
        "dtype": dtype_contract(array)[0],
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": list(array.shape),
        "zarr_format": 2,
    }


def write_array_chunks(path: Path, array: NDArray[np.generic], rows: int) -> None:
    path.mkdir()
    chunks = (rows, *array.shape[1:])
    (path / ".zarray").write_bytes(_json(_array_metadata(array, chunks)))
    for start in range(0, array.shape[0], rows):
        index = start // rows
        suffix = ".".join([str(index), *("0" for _ in array.shape[1:])])
        chunk = np.zeros(chunks, dtype=array.dtype)
        count = min(rows, array.shape[0] - start)
        chunk[:count] = array[start : start + count]
        (path / suffix).write_bytes(chunk.tobytes())


def canonical_digest(
    arrays: dict[str, PaperArray], episode_ends: NDArray[np.int64], episode_ids: list[str]
) -> str:
    """Hash the exact persisted arrays, boundaries, and ordered external episode IDs."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        values = np.ascontiguousarray(arrays[name].values)
        digest.update(_json([name, dtype_contract(values)[1], list(values.shape)]))
        digest.update(values.tobytes())
    digest.update(_json(["episode_ends", "int64", list(episode_ends.shape)]))
    digest.update(np.ascontiguousarray(episode_ends).tobytes())
    digest.update(_json(["episode_ids", episode_ids]))
    return digest.hexdigest()


def _array_record(array: PaperArray) -> dict[str, object]:
    values = np.ascontiguousarray(array.values)
    return {
        "dtype": dtype_contract(values)[1],
        "shape": list(values.shape),
        "unit": array.unit,
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def validate_native_arrays(
    arrays: dict[str, PaperArray], episode_ends: NDArray[np.int64], episode_ids: list[str]
) -> None:
    if set(arrays) != set(ARRAY_NAMES):
        raise PaperViewError("native arrays must contain the exact policy and alignment keys")
    values = {name: np.ascontiguousarray(arrays[name].values) for name in ARRAY_NAMES}
    row_count = values["cam_top"].shape[0]
    if _SINGLE_CAM:
        expected = {
            "cam_top": ((row_count, 96, 96, 3), np.dtype(np.uint8)),
            "agent_pos": ((row_count, 5), np.dtype(np.float32)),
            "action": ((row_count, 2), np.dtype(np.float32)),
            "timestamp": ((row_count,), np.dtype(np.float64)),
            "episode_id": ((row_count,), np.dtype(np.int64)),
            "frame_index": ((row_count,), np.dtype(np.int64)),
        }
    else:
        expected = {
            "cam_top": ((row_count, 224, 224, 3), np.dtype(np.uint8)),
            "cam_side": ((row_count, 224, 224, 3), np.dtype(np.uint8)),
            "agent_pos": ((row_count, 5), np.dtype(np.float32)),
            "action": ((row_count, 2), np.dtype(np.float32)),
            "timestamp": ((row_count,), np.dtype(np.float64)),
            "episode_id": ((row_count,), np.dtype(np.int64)),
            "frame_index": ((row_count,), np.dtype(np.int64)),
        }
    if row_count == 0 or any(
        values[name].shape != shape or values[name].dtype != dtype
        for name, (shape, dtype) in expected.items()
    ):
        raise PaperViewError("native array shape or dtype mismatch")
    if not np.isfinite(values["agent_pos"]).all() or not np.isfinite(values["action"]).all():
        raise PaperViewError("non-finite native state/action")
    action_values = values["action"].astype(np.float32, copy=False)
    if bool((action_values < -1.0).any()) or bool((action_values > 1.0).any()):
        raise PaperViewError("native action must be within [-1,1]")
    if not np.isfinite(values["timestamp"]).all():
        raise PaperViewError("non-finite native timestamp")
    if (
        len(episode_ids) == 0
        or len(episode_ids) != len(set(episode_ids))
        or any(not value for value in episode_ids)
    ):
        raise PaperViewError("episode IDs are empty or duplicate")
    ends = episode_ends.tolist()
    if (
        episode_ends.dtype != np.dtype(np.int64)
        or episode_ends.shape != (len(episode_ids),)
        or ends[-1] != row_count
        or any(right <= left for left, right in zip([0, *ends[:-1]], ends, strict=True))
    ):
        raise PaperViewError("episode boundaries are invalid")
    expected_episode = np.empty(row_count, dtype=np.int64)
    expected_frame = np.empty(row_count, dtype=np.int64)
    start = 0
    for ordinal, end in enumerate(ends):
        expected_episode[start:end] = ordinal
        expected_frame[start:end] = np.arange(end - start, dtype=np.int64)
        start = end
    if (
        values["episode_id"].tolist() != expected_episode.tolist()
        or values["frame_index"].tolist() != expected_frame.tolist()
    ):
        raise PaperViewError("episode/frame alignment is not chronological")
    expected_time = expected_frame.astype(np.float64) / FPS
    if values["timestamp"].tolist() != expected_time.tolist():
        raise PaperViewError("timestamp must equal frame_index/10 exactly")


def write_paper_view(
    destination: Path,
    arrays: dict[str, PaperArray],
    episode_ends: NDArray[np.int64],
    metadata: PaperViewMetadata,
) -> Path:
    """Validate all bytes, then atomically publish a native dual-camera view."""
    validate_native_arrays(arrays, episode_ends, metadata.episode_ids)
    expected_canonical = canonical_digest(arrays, episode_ends, metadata.episode_ids)
    supplied_canonical = require_sha256(metadata.canonical_digest, "canonical digest")
    if supplied_canonical != expected_canonical:
        raise PaperViewError("canonical digest mismatch")
    root_provenance = validate_root_provenance(
        metadata.root_provenance, metadata.episode_ids, episode_ends
    )
    supplied_root = require_sha256(metadata.root_digest, "root digest")
    if supplied_root != root_provenance_digest(root_provenance):
        raise PaperViewError("root provenance digest mismatch")
    supplied_runtime = require_sha256(metadata.runtime_lock_digest, "runtime lock digest")
    if supplied_runtime != trusted_runtime_lock_digest():
        raise PaperViewError("trusted runtime lock digest mismatch")
    destination = safe_paper_path(destination)
    if destination.exists():
        raise FileExistsError(f"immutable native view already exists: {destination}")
    if destination.parent.exists() and any(destination.parent.glob(f".{destination.name}.tmp-*")):
        raise PaperViewError(f"destination-related staging state exists: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    staging.mkdir()
    try:
        (staging / ".zgroup").write_bytes(_json({"zarr_format": 2}))
        data = staging / "data"
        data.mkdir()
        (data / ".zgroup").write_bytes(_json({"zarr_format": 2}))
        for name in ARRAY_NAMES:
            write_array_chunks(data / name, arrays[name].values, CHUNK_ROWS[name])
        write_array_chunks(staging / "episode_ends", episode_ends, CHUNK_ROWS["episode_ends"])
        (staging / "splits.json").write_bytes(_json(metadata.splits))
        records = {name: _array_record(arrays[name]) for name in sorted(arrays)}
        records["episode_ends"] = _array_record(PaperArray(episode_ends, "cumulative frames"))
        manifest: dict[str, object] = {
            "arrays": records,
            "canonical_digest": metadata.canonical_digest,
            "contract_schema": "pusht-so100-native-v1",
            "episode_ids": metadata.episode_ids,
            "exporter_revision": EXPORTER_REVISION,
            "fps": FPS,
            "root_digest": metadata.root_digest,
            "root_provenance": root_provenance,
            "runtime_lock_digest": metadata.runtime_lock_digest,
            "splits": metadata.splits,
            "training_eligible": metadata.training_eligible,
            "zarr_format": 2,
        }
        (staging / "manifest.json").write_bytes(_json(manifest))
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def load_paper_view(path: Path) -> LoadedPaperView:
    from .paper_view_reader import load_paper_view as load

    return load(path)


def validate_training_view(path: Path) -> LoadedPaperView:
    from .paper_view_reader import validate_training_view as validate

    return validate(path)
