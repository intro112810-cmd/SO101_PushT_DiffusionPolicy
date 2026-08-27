"""Strict reload and verification for immutable native pushT-so100 views."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .paper_view import (
    ARRAY_NAMES,
    CHUNK_ROWS,
    EXPORTER_REVISION,
    FPS,
    LoadedPaperView,
    PaperArray,
    PaperViewError,
    canonical_digest,
    dtype_contract,
    require_sha256,
    root_provenance_digest,
    safe_paper_path,
    trusted_runtime_lock_digest,
    validate_native_arrays,
    validate_root_provenance,
)

_MANIFEST_KEYS = {
    "arrays",
    "canonical_digest",
    "contract_schema",
    "episode_ids",
    "exporter_revision",
    "fps",
    "root_digest",
    "root_provenance",
    "runtime_lock_digest",
    "splits",
    "training_eligible",
    "zarr_format",
}
_SINGLE_CAM = os.environ.get("PUSHT_SINGLE_CAM") == "1"

if _SINGLE_CAM:
    _EXPECTED_DTYPES = {
        "cam_top": "|u1",
        "agent_pos": "<f4",
        "action": "<f4",
        "timestamp": "<f8",
        "episode_id": "<i8",
        "frame_index": "<i8",
        "episode_ends": "<i8",
    }
else:
    _EXPECTED_DTYPES = {
        "cam_top": "|u1",
        "cam_side": "|u1",
        "agent_pos": "<f4",
        "action": "<f4",
        "timestamp": "<f8",
        "episode_id": "<i8",
        "frame_index": "<i8",
        "episode_ends": "<i8",
    }


def _object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperViewError(f"invalid JSON metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise PaperViewError(f"JSON object required: {path.name}")
    return cast("dict[str, object]", value)


def _read_array(path: Path) -> tuple[NDArray[np.generic], set[Path]]:
    metadata = _object(path / ".zarray")
    if set(metadata) != {
        "chunks",
        "compressor",
        "dtype",
        "fill_value",
        "filters",
        "order",
        "shape",
        "zarr_format",
    }:
        raise PaperViewError(f"malformed Zarr metadata: {path.name}")
    if (
        metadata.get("zarr_format") != 2
        or metadata.get("compressor") is not None
        or metadata.get("filters") is not None
        or metadata.get("order") != "C"
    ):
        raise PaperViewError(f"unsupported Zarr metadata: {path.name}")
    raw_shape = metadata.get("shape")
    raw_chunks = metadata.get("chunks")
    raw_dtype = metadata.get("dtype")
    if (
        not isinstance(raw_shape, list)
        or not isinstance(raw_chunks, list)
        or not isinstance(raw_dtype, str)
    ):
        raise PaperViewError(f"malformed Zarr metadata: {path.name}")
    dimensions = [*cast("list[object]", raw_shape), *cast("list[object]", raw_chunks)]
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in dimensions
    ):
        raise PaperViewError(f"invalid Zarr dimensions: {path.name}")
    shape = tuple(cast("list[int]", raw_shape))
    chunks = tuple(cast("list[int]", raw_chunks))
    descriptors = {
        "|u1": np.dtype(np.uint8),
        "<f4": np.dtype(np.float32),
        "<f8": np.dtype(np.float64),
        "<i8": np.dtype(np.int64),
    }
    dtype = descriptors.get(raw_dtype)
    expected_rows = CHUNK_ROWS.get(path.name)
    if (
        dtype is None
        or raw_dtype != _EXPECTED_DTYPES.get(path.name)
        or expected_rows is None
        or chunks != (expected_rows, *shape[1:])
        or metadata.get("fill_value") != 0
    ):
        raise PaperViewError(f"unsupported Zarr dtype or chunk contract: {path.name}")
    output = np.empty(shape, dtype=dtype)
    members = {path / ".zarray"}
    for start in range(0, shape[0], chunks[0]):
        index = start // chunks[0]
        chunk = path / ".".join([str(index), *("0" for _ in shape[1:])])
        members.add(chunk)
        try:
            encoded = chunk.read_bytes()
        except OSError as exc:
            raise PaperViewError(f"missing native chunk: {path.name}") from exc
        expected_size = int(np.prod(chunks, dtype=np.int64)) * dtype.itemsize
        if len(encoded) != expected_size:
            raise PaperViewError(f"invalid chunk size: {path.name}")
        values = cast("NDArray[np.generic]", np.ndarray(chunks, dtype=dtype, buffer=encoded))
        count = min(chunks[0], shape[0] - start)
        output[start : start + count] = values[:count]
    return output, members


def _load_paper_view(path: Path) -> LoadedPaperView:
    root = safe_paper_path(path)
    if not root.is_dir() or root.is_symlink():
        raise PaperViewError("native view is not a real directory")
    manifest = _object(root / "manifest.json")
    splits = _object(root / "splits.json")
    if set(manifest) != _MANIFEST_KEYS:
        raise PaperViewError("native manifest keys are malformed")
    if _object(root / ".zgroup") != {"zarr_format": 2} or _object(root / "data/.zgroup") != {
        "zarr_format": 2
    }:
        raise PaperViewError("Zarr group metadata contract mismatch")
    supplied_canonical = require_sha256(manifest.get("canonical_digest"), "canonical digest")
    supplied_root = require_sha256(manifest.get("root_digest"), "root digest")
    supplied_runtime = require_sha256(manifest.get("runtime_lock_digest"), "runtime lock digest")
    if supplied_runtime != trusted_runtime_lock_digest():
        raise PaperViewError("trusted runtime lock digest mismatch")
    if (
        manifest.get("zarr_format") != 2
        or manifest.get("exporter_revision") != EXPORTER_REVISION
        or manifest.get("contract_schema") != "pusht-so100-native-v1"
        or manifest.get("fps") != FPS
        or manifest.get("splits") != splits
    ):
        raise PaperViewError("native manifest contract mismatch")
    expected_files = {
        root / ".zgroup",
        root / "manifest.json",
        root / "splits.json",
        root / "data/.zgroup",
    }
    arrays: dict[str, NDArray[np.generic]] = {}
    records_raw = manifest.get("arrays")
    if not isinstance(records_raw, dict) or set(cast("dict[object, object]", records_raw)) != {
        *ARRAY_NAMES,
        "episode_ends",
    }:
        raise PaperViewError("native array manifest keys are not exact")
    records = cast("dict[str, object]", records_raw)
    for name in ARRAY_NAMES:
        arrays[name], members = _read_array(root / "data" / name)
        expected_files.update(members)
    ends_raw, members = _read_array(root / "episode_ends")
    expected_files.update(members)
    if ends_raw.dtype != np.dtype(np.int64):
        raise PaperViewError("episode boundaries dtype mismatch")
    ends = cast("NDArray[np.int64]", ends_raw)
    episode_ids_raw = manifest.get("episode_ids")
    if not isinstance(episode_ids_raw, list) or not all(
        isinstance(value, str) for value in cast("list[object]", episode_ids_raw)
    ):
        raise PaperViewError("episode IDs are malformed")
    episode_ids = cast("list[str]", episode_ids_raw)
    root_provenance = validate_root_provenance(manifest.get("root_provenance"), episode_ids, ends)
    if supplied_root != root_provenance_digest(root_provenance):
        raise PaperViewError("root provenance digest mismatch")
    validate_native_arrays(
        {name: _paper_array(arrays[name], records[name]) for name in ARRAY_NAMES},
        ends,
        episode_ids,
    )
    paper_arrays = {name: _paper_array(arrays[name], records[name]) for name in ARRAY_NAMES}
    for name, values in {**arrays, "episode_ends": ends}.items():
        record = records.get(name)
        if not isinstance(record, dict):
            raise PaperViewError(f"array record is malformed: {name}")
        typed = cast("dict[str, object]", record)
        if set(typed) != {"dtype", "shape", "unit", "sha256"} or (
            typed.get("shape") != list(values.shape)
            or typed.get("dtype") != dtype_contract(values)[1]
            or typed.get("sha256") != hashlib.sha256(values.tobytes()).hexdigest()
        ):
            raise PaperViewError(f"array hash or metadata mismatch: {name}")
    if supplied_canonical != canonical_digest(paper_arrays, ends, episode_ids):
        raise PaperViewError("canonical digest mismatch")
    actual_files: set[Path] = set()
    for member in root.rglob("*"):
        mode = member.lstat().st_mode
        if stat.S_ISLNK(mode) or (not stat.S_ISDIR(mode) and not stat.S_ISREG(mode)):
            raise PaperViewError("special native view member is forbidden")
        if stat.S_ISREG(mode):
            actual_files.add(member)
    if actual_files != expected_files:
        raise PaperViewError("native view membership mismatch")
    return LoadedPaperView(arrays, ends, manifest, splits)


def _paper_array(values: NDArray[np.generic], record: object) -> PaperArray:
    if not isinstance(record, dict):
        raise PaperViewError("array unit metadata is malformed")
    typed = cast("dict[str, object]", record)
    if not isinstance(typed.get("unit"), str):
        raise PaperViewError("array unit metadata is malformed")
    return PaperArray(values, cast(str, typed["unit"]))


def load_paper_view(path: Path) -> LoadedPaperView:
    """Reload and fully verify an immutable native view."""
    try:
        return _load_paper_view(path)
    except PaperViewError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise PaperViewError("malformed native view") from exc


def validate_training_view(path: Path) -> LoadedPaperView:
    loaded = load_paper_view(path)
    if (
        loaded.manifest.get("training_eligible") is not True
        or loaded.splits.get("frozen") is not True
    ):
        raise PaperViewError("native view is not frozen and training eligible")
    try:
        from decimal import Decimal

        from .splits import (
            ExperimentConfig,
            SplitManifest,
            validate_split_manifest,
        )

        manifest = SplitManifest.from_dict(loaded.splits)
        config = ExperimentConfig(
            "pusht-so100-experiment-v1",
            manifest.target_episode_count,
            {
                "train": Decimal(manifest.train_ratio),
                "validation": Decimal(manifest.validation_ratio),
                "test": Decimal(manifest.test_ratio),
            },
        )
        episode_ids = cast("list[str]", loaded.manifest["episode_ids"])
        validate_split_manifest(
            manifest,
            episode_ids,
            config,
            source_digest=manifest.source_digest,
        )
    except (ValueError, ArithmeticError) as exc:
        raise PaperViewError(f"invalid frozen split manifest: {exc}") from exc
    return loaded
