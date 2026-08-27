"""Fail-closed LeRobot 0.4.4 importer for native multi-episode pushT-so100 data."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat
import sys
from typing import cast

import numpy as np
from numpy.typing import NDArray
import pyarrow as pa
import pyarrow.parquet as pq

from .paper_view import (
    PaperArray,
    PaperViewMetadata,
    canonical_digest,
    root_provenance_digest,
    trusted_runtime_lock_digest,
    write_paper_view,
)

CAM_TOP_KEY = "observation.images.cam_top"
CAM_SIDE_KEY = "observation.images.cam_side"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
FPS = 10
_POLICY_FEATURES = {
    CAM_TOP_KEY: ("video", [224, 224, 3]),
    CAM_SIDE_KEY: ("video", [224, 224, 3]),
    STATE_KEY: ("float32", [5]),
    ACTION_KEY: ("float32", [2]),
}
_SYSTEM_FEATURES = {
    "timestamp": ("float32", [1]),
    "frame_index": ("int64", [1]),
    "episode_index": ("int64", [1]),
    "index": ("int64", [1]),
    "task_index": ("int64", [1]),
}
_DATA_COLUMNS = [
    STATE_KEY,
    ACTION_KEY,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]


class ImporterError(ValueError):
    """Raised when a source dataset violates the exact native contract."""


def _object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.lstat().st_mode):
        raise ImporterError(f"missing or unsafe {label}: {path}")
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImporterError(f"malformed {label}") from exc
    if not isinstance(value, dict):
        raise ImporterError(f"malformed {label}: object required")
    return cast("dict[str, object]", value)


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ImporterError(f"malformed metadata: {label}")
    return value


def _validate_info(repo: Path) -> tuple[dict[str, object], int, int]:
    info = _object(repo / "meta/info.json", "LeRobot info.json")
    if info.get("codebase_version") != "v3.0" or info.get("fps") != FPS:
        raise ImporterError("LeRobot metadata must be v3.0 at FPS 10")
    total_episodes = _positive_int(info.get("total_episodes"), "total_episodes")
    total_frames = _positive_int(info.get("total_frames"), "total_frames")
    if (
        info.get("data_path") != "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        or info.get("video_path")
        != "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    ):
        raise ImporterError("malformed LeRobot data/video path metadata")
    raw_features = info.get("features")
    if not isinstance(raw_features, dict):
        raise ImporterError("malformed LeRobot feature metadata")
    features = cast("dict[str, object]", raw_features)
    expected = {**_POLICY_FEATURES, **_SYSTEM_FEATURES}
    if set(features) != set(expected):
        missing = sorted(set(expected) - set(features))
        extra = sorted(set(features) - set(expected))
        detail = missing[0] if missing else extra[0]
        raise ImporterError(f"native policy feature keys are not exact: {detail}")
    for key, (dtype, shape) in expected.items():
        descriptor = features[key]
        if not isinstance(descriptor, dict):
            raise ImporterError(f"malformed feature metadata: {key}")
        typed = cast("dict[str, object]", descriptor)
        if typed.get("dtype") != dtype or typed.get("shape") != shape:
            raise ImporterError(f"native feature contract mismatch: {key}")
    return info, total_episodes, total_frames


def _read_tables(paths: list[Path], label: str) -> pa.Table:
    if not paths:
        raise ImporterError(f"missing {label} parquet")
    try:
        return pa.concat_tables([pq.read_table(path) for path in paths], promote_options="default")
    except (OSError, pa.ArrowException) as exc:
        raise ImporterError(f"malformed {label} parquet") from exc


def _validate_data_schema(table: pa.Table) -> None:
    if table.column_names != _DATA_COLUMNS:
        raise ImporterError("data parquet columns are not exact")
    schema = table.schema
    state_type = schema.field(STATE_KEY).type
    action_type = schema.field(ACTION_KEY).type
    if not (
        pa.types.is_fixed_size_list(state_type)
        and state_type.list_size == 5
        and pa.types.is_float32(state_type.value_type)
        and pa.types.is_fixed_size_list(action_type)
        and action_type.list_size == 2
        and pa.types.is_float32(action_type.value_type)
        and pa.types.is_float32(schema.field("timestamp").type)
        and all(pa.types.is_int64(schema.field(name).type) for name in _DATA_COLUMNS[3:])
    ):
        raise ImporterError("data parquet dtype/shape contract mismatch")


def _episode_rows(repo: Path, total_episodes: int) -> list[dict[str, object]]:
    paths = sorted((repo / "meta/episodes").glob("chunk-*/file-*.parquet"))
    table = _read_tables(paths, "episode metadata")
    rows = cast("list[dict[str, object]]", table.to_pylist())
    ids = [row.get("episode_index") for row in rows]
    if len(rows) != total_episodes or ids != list(range(total_episodes)):
        raise ImporterError("duplicate, missing, or nonchronological episode IDs")
    return rows


def _declared_path(repo: Path, prefix: str, row: dict[str, object], suffix: str) -> Path:
    chunk = row.get(f"{prefix}/chunk_index")
    file_index = row.get(f"{prefix}/file_index")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (chunk, file_index)
    ):
        raise ImporterError(f"malformed episode {prefix} location metadata")
    return repo / prefix / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.{suffix}"


def decode_video(path: Path) -> NDArray[np.uint8]:
    if path.is_symlink() or not path.is_file():
        raise ImporterError(f"missing camera video: {path}")
    try:
        import av

        container = av.open(str(path))
        try:
            if len(container.streams.video) != 1:
                raise ImporterError(f"camera video must have one stream: {path}")
            stream = container.streams.video[0]
            if stream.average_rate is None or stream.average_rate != FPS:
                raise ImporterError(f"camera video FPS must be 10: {path}")
            if stream.width != 224 or stream.height != 224:
                raise ImporterError(f"camera video shape must be 224x224: {path}")
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        finally:
            container.close()
    except ImporterError:
        raise
    except Exception as exc:
        raise ImporterError(f"cannot decode camera video: {path}") from exc
    if not frames:
        raise ImporterError(f"camera video is empty: {path}")
    values = np.stack(frames)
    if values.dtype != np.dtype(np.uint8) or values.shape[1:] != (224, 224, 3):
        raise ImporterError(f"decoded camera contract mismatch: {path}")
    return values


def _video_slice(
    row: dict[str, object], key: str, decoded: dict[Path, NDArray[np.uint8]], repo: Path
) -> NDArray[np.uint8]:
    prefix = f"videos/{key}"
    path = _declared_path(repo, prefix, row, "mp4")
    if path not in decoded:
        decoded[path] = decode_video(path)
    start_raw = row.get(f"{prefix}/from_timestamp")
    end_raw = row.get(f"{prefix}/to_timestamp")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (start_raw, end_raw)
    ):
        raise ImporterError(f"malformed camera timestamp metadata: {key}")
    start_value, end_value = float(cast(float, start_raw)), float(cast(float, end_raw))
    start, end = round(start_value * FPS), round(end_value * FPS)
    if (
        not math.isclose(start_value, start / FPS, abs_tol=1e-9)
        or not math.isclose(end_value, end / FPS, abs_tol=1e-9)
        or start < 0
        or end <= start
        or end > decoded[path].shape[0]
    ):
        raise ImporterError(f"partial or malformed camera episode: {key}")
    return decoded[path][start:end]


def _source_members(repo: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    for path in sorted(repo.rglob("*")):
        if path.is_symlink():
            raise ImporterError(f"source dataset contains symlink: {path}")
        if path.is_file():
            members[path.relative_to(repo).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    if not members:
        raise ImporterError("source dataset has no provenance members")
    return members


def _explicit_nonproduction_fixture(repo: Path) -> bool:
    marker = repo / "synthetic-fixture.NON_PRODUCTION.json"
    if not marker.exists():
        return False
    value = _object(marker, "non-production fixture marker")
    if value != {
        "schema": 1,
        "artifact_type": "synthetic_fixture",
        "production_eligible": False,
        "comparison_eligible": False,
        "reason": "synthetic_fixture_not_comparison_eligible",
    }:
        raise ImporterError("malformed non-production fixture marker")
    return True


def _load_native_arrays(repo: Path) -> tuple[dict[str, PaperArray], NDArray[np.int64], list[str]]:
    _, total_episodes, total_frames = _validate_info(repo)
    episode_rows = _episode_rows(repo, total_episodes)
    data_paths = sorted((repo / "data").glob("chunk-*/file-*.parquet"))
    declared_data_paths = {_declared_path(repo, "data", row, "parquet") for row in episode_rows}
    if set(data_paths) != declared_data_paths:
        raise ImporterError("data file membership does not match episode metadata")
    table = _read_tables(data_paths, "data")
    _validate_data_schema(table)
    if table.num_rows != total_frames:
        raise ImporterError("data row count does not match metadata")
    global_indices = cast("list[int]", table["index"].to_pylist())
    if global_indices != list(range(total_frames)):
        raise ImporterError("data rows are not chronological")
    state_rows = cast("list[list[float]]", table[STATE_KEY].to_pylist())
    action_rows = cast("list[list[float]]", table[ACTION_KEY].to_pylist())
    states = np.asarray(state_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.float32)
    if states.dtype != np.dtype(np.float32) or actions.dtype != np.dtype(np.float32):
        raise ImporterError("state/action dtype contract mismatch")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ImporterError("non-finite state/action")
    if bool((actions < -1.0).any()) or bool((actions > 1.0).any()):
        raise ImporterError("action is outside [-1,1]")
    cameras: dict[str, dict[Path, NDArray[np.uint8]]] = {CAM_TOP_KEY: {}, CAM_SIDE_KEY: {}}
    top_parts: list[NDArray[np.uint8]] = []
    side_parts: list[NDArray[np.uint8]] = []
    state_parts: list[NDArray[np.float32]] = []
    action_parts: list[NDArray[np.float32]] = []
    episode_ends: list[int] = []
    episode_values: list[int] = []
    frame_values: list[int] = []
    cursor = 0
    for ordinal, row in enumerate(episode_rows):
        length = _positive_int(row.get("length"), f"episode {ordinal} length")
        start = row.get("dataset_from_index")
        end = row.get("dataset_to_index")
        if start != cursor or end != cursor + length:
            raise ImporterError(f"partial or nonchronological episode: {ordinal}")
        row_episode = cast("list[int]", table["episode_index"].slice(cursor, length).to_pylist())
        row_frames = cast("list[int]", table["frame_index"].slice(cursor, length).to_pylist())
        row_timestamps = cast("list[float]", table["timestamp"].slice(cursor, length).to_pylist())
        if row_episode != [ordinal] * length or row_frames != list(range(length)):
            raise ImporterError(f"episode/frame alignment mismatch: {ordinal}")
        if not all(
            # LeRobot persists timestamps as float32, so frame/FPS values above ~32
        # carry float32 rounding error up to ~2e-6 (e.g. 32.1 -> 32.099998).
        # 1e-4 keeps the exact frame/FPS contract while accepting real float32
        # storage rounding; anything further off is a genuine timestamp defect.
        math.isclose(float(value), frame / FPS, abs_tol=1e-4)
            for frame, value in enumerate(row_timestamps)
        ):
            raise ImporterError(f"source frame timestamps are malformed: {ordinal}")
        top = _video_slice(row, CAM_TOP_KEY, cameras[CAM_TOP_KEY], repo)
        side = _video_slice(row, CAM_SIDE_KEY, cameras[CAM_SIDE_KEY], repo)
        if top.shape[0] != length or side.shape[0] != length:
            raise ImporterError(f"camera/data row count mismatch: {ordinal}")
        top_parts.append(top)
        side_parts.append(side)
        state_parts.append(states[cursor : cursor + length])
        action_parts.append(actions[cursor : cursor + length])
        episode_values.extend([ordinal] * length)
        frame_values.extend(range(length))
        cursor += length
        episode_ends.append(cursor)
    if cursor != total_frames:
        raise ImporterError("partial final episode")
    for key, decoded in cameras.items():
        actual = set((repo / "videos" / key).glob("chunk-*/file-*.mp4"))
        if actual != set(decoded):
            raise ImporterError(f"camera file membership mismatch: {key}")
        if sum(video.shape[0] for video in decoded.values()) != total_frames:
            raise ImporterError(f"camera/data total row count mismatch: {key}")
    arrays = {
        "cam_top": PaperArray(np.concatenate(top_parts), "rgb intensity"),
        "cam_side": PaperArray(np.concatenate(side_parts), "rgb intensity"),
        "agent_pos": PaperArray(np.concatenate(state_parts), "radians"),
        "action": PaperArray(np.concatenate(action_parts), "absolute normalized mocap XY"),
        "timestamp": PaperArray(np.asarray(frame_values, dtype=np.float64) / FPS, "seconds"),
        "episode_id": PaperArray(np.asarray(episode_values, dtype=np.int64), "episode ordinal"),
        "frame_index": PaperArray(np.asarray(frame_values, dtype=np.int64), "frame ordinal"),
    }
    return (
        arrays,
        np.asarray(episode_ends, dtype=np.int64),
        [str(value) for value in range(total_episodes)],
    )


def import_repo_store(repo: Path, output: Path) -> int:
    """Validate every source episode and atomically persist one native dual-camera view."""
    try:
        if output.exists():
            raise ImporterError(f"output already exists: {output}")
        arrays, episode_ends, episode_ids = _load_native_arrays(repo)
        fixture = _explicit_nonproduction_fixture(repo)
        persisted_digest = canonical_digest(arrays, episode_ends, episode_ids)
        starts = [0, *episode_ends.tolist()[:-1]]
        root_provenance: dict[str, object] = {
            "schema": "pusht-so100-root-provenance-v1",
            "source_members": _source_members(repo),
            "episodes": [
                {"episode_id": episode_id, "length": end - start}
                for episode_id, start, end in zip(
                    episode_ids, starts, episode_ends.tolist(), strict=True
                )
            ],
        }
        source_digest = root_provenance_digest(root_provenance)
        lock_digest = trusted_runtime_lock_digest()
        splits: dict[str, object] = {
            "frozen": False,
            "training_eligible": False,
            "reason": (
                "synthetic_fixture_not_comparison_eligible"
                if fixture
                else "split_manifest_not_frozen"
            ),
            "train": [],
            "validation": [],
            "test": [],
        }
        write_paper_view(
            output,
            arrays,
            episode_ends,
            PaperViewMetadata(
                persisted_digest,
                source_digest,
                root_provenance,
                episode_ids,
                splits,
                lock_digest,
                False,
            ),
        )
    except Exception as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"episodes": episode_ids, "frames": int(episode_ends[-1]), "store": str(output)},
            sort_keys=True,
        )
    )
    return 0
