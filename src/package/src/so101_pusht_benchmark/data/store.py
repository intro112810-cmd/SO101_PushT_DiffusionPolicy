"""Atomic local-only LeRobot v3 dataset persistence and immutable raw attempts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from importlib import import_module
from pathlib import Path
import stat
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from ..workspace import WorkspacePolicyError, runtime_artifact_root

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


Provenance = dict[str, object]


class CurrentManifest(TypedDict):
    dataset_id: str
    version: str
    attempt_ids: list[str]
    files: dict[str, str]
    provenance: Provenance
    root_digest: str


class _LeRobotDatasetType(Protocol):
    create: Callable[..., object]

    def __call__(self, **kwargs: object) -> object: ...


class _LeRobotModule(Protocol):
    LeRobotDataset: _LeRobotDatasetType


class DatasetProtocol(Protocol):
    def add_frame(self, frame: dict[str, object]) -> None: ...
    def save_episode(self) -> None: ...
    def finalize(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PublishRequest:
    attempt_id: str
    frames: Sequence[FrameRecord]
    metadata: dict[str, object]
    features: dict[str, object]
    create: Callable[..., DatasetProtocol] | None = None
    reload: Callable[[Path], None] | None = None
    fault: Callable[[str], None] | None = None


@dataclass(frozen=True, slots=True)
class FrameRecord:
    frame_index: int
    timestamp: float
    observation: dict[str, NDArray[np.generic]]
    action: tuple[float, float, float]
    raw_axes: object | None
    requested_target: tuple[float, float, float]
    telemetry: dict[str, object]
    next_observation: dict[str, NDArray[np.generic]]
    applied: bool


def _safe_id(value: str, label: str) -> str:
    if not value or not value.replace("_", "a").replace("-", "a").isalnum():
        raise ValueError(f"unsafe {label}")
    return value


def _safe_root(path: Path) -> Path:
    root = runtime_artifact_root().resolve()
    absolute = path.absolute()
    for parent in (absolute, *absolute.parents):
        if parent.exists() and stat.S_ISLNK(parent.lstat().st_mode):
            raise WorkspacePolicyError(f"symlink path is forbidden: {path}")
        if parent == root.parent:
            break
    resolved = absolute.resolve()
    if resolved == root or root not in resolved.parents:
        raise WorkspacePolicyError(f"dataset root must be beneath artifact root: {path}")
    return resolved


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, sort_keys=True, separators=(",", ":")))


def _atomic_text(path: Path, encoded: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _digest_tree(root: Path) -> tuple[str, dict[str, str]]:
    """Hash regular files while allowing ordinary directories as containers."""
    files: dict[str, str] = {}
    if root.is_symlink() or not root.is_dir():
        raise WorkspacePolicyError(f"canonical root is not a real directory: {root}")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise WorkspacePolicyError(f"canonical tree has symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise WorkspacePolicyError(f"canonical tree has unsafe entry: {relative}")
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), files


digest_tree = _digest_tree


class LocalDatasetStore:
    """Stores immutable rejected evidence and atomically publishes accepted local datasets."""

    def __init__(self, root: Path, repo_id: str = "local/so101-pusht-pilot") -> None:
        self.root = _safe_root(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.raw = self.root / "rejected" / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        (self.root / "versions").mkdir(exist_ok=True)
        (self.root / "staging").mkdir(exist_ok=True)

    def recover(self) -> dict[str, list[str]]:
        """Quarantine incomplete transactions without changing a valid current pointer."""
        orphaned: list[str] = []
        rejected = self.root / "rejected" / "staging"
        rejected.mkdir(parents=True, exist_ok=True)
        for journal in sorted((self.root / "staging").glob("*.journal.json")):
            attempt_id = journal.name.removesuffix(".journal.json")
            staging = self.root / "staging" / attempt_id
            if staging.exists():
                destination = rejected / attempt_id
                if destination.exists():
                    raise FileExistsError(f"recovery destination exists: {attempt_id}")
                staging.replace(destination)
            journal.replace(rejected / journal.name)
            orphaned.append(attempt_id)
        return {"orphaned": orphaned, "current": self._current_attempts()}

    def write_attempt(
        self, attempt_id: str, frames: Sequence[FrameRecord], metadata: dict[str, object]
    ) -> Path:
        _safe_id(attempt_id, "attempt id")
        target = self.raw / f"{attempt_id}.json"
        if target.exists():
            raise FileExistsError(f"duplicate attempt id: {attempt_id}")
        payload: dict[str, object] = {
            "metadata": metadata,
            "frames": [self._raw_frame(frame) for frame in frames],
        }
        payload["sha256"] = self.payload_digest(payload)
        _atomic_json(target, payload)
        return target

    @staticmethod
    def _raw_frame(frame: FrameRecord) -> dict[str, object]:
        return {
            "frame_index": frame.frame_index,
            "timestamp": frame.timestamp,
            "action": list(frame.action),
            "observation": {key: value.tolist() for key, value in frame.observation.items()},
            "next_observation": {
                key: value.tolist() for key, value in frame.next_observation.items()
            },
            "applied": frame.applied,
            "telemetry": {
                "raw_axes": frame.raw_axes,
                "requested_target": frame.requested_target,
                **frame.telemetry,
            },
        }

    @staticmethod
    def payload_digest(payload: dict[str, object]) -> str:
        copied = {key: value for key, value in payload.items() if key != "sha256"}
        return hashlib.sha256(
            json.dumps(copied, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def publish_human_episode(self, request: PublishRequest) -> Path:
        """Build, reload-check, digest-address, then publish one episode."""
        attempt_id = request.attempt_id
        frames = request.frames
        metadata = request.metadata
        features = request.features
        _safe_id(attempt_id, "attempt id")
        if (self.root / "current.json").exists() and attempt_id in self._current_attempts():
            raise FileExistsError(f"duplicate accepted attempt id: {attempt_id}")
        self._require_human_metadata(metadata)
        staging = self.root / "staging" / attempt_id
        if staging.exists():
            raise FileExistsError(f"staging exists; recover or reject it first: {attempt_id}")
        # The staging parent exists, but LeRobot's exact root must be new.  This
        # is also the boundary that prevents a create call from overwriting data.
        _atomic_json(
            self.root / "staging" / f"{attempt_id}.journal.json",
            {"attempt_id": attempt_id, "state": "intent"},
        )
        task = metadata.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("task is mandatory for every canonical frame")
        self._validate_features(features)
        try:
            maker = request.create or self._lerobot_create
            dataset = maker(repo_id=self.repo_id, root=staging, fps=10, features=features)
            for frame in frames:
                dataset.add_frame(self._dataset_frame(frame, task))
            dataset.save_episode()
            if request.fault is not None:
                request.fault("after_save_episode")
            dataset.finalize()
            if request.fault is not None:
                request.fault("after_finalize")
            (request.reload or self._reload_local)(staging)
            content_digest, content_files = _digest_tree(staging)
            _atomic_json(
                staging / "version_manifest.json",
                {
                    "attempt_id": attempt_id,
                    "digest": content_digest,
                    "files": content_files,
                    "features": features,
                    "task": task,
                },
            )
            root_digest, files = _digest_tree(staging)
            destination = self.root / "versions" / content_digest
            if destination.exists():
                raise FileExistsError(f"duplicate dataset digest: {content_digest}")
            if request.fault is not None:
                request.fault("before_version_rename")
            staging.replace(destination)
            self._write_current(attempt_id, content_digest, root_digest, files, metadata)
            journal = self.root / "staging" / f"{attempt_id}.journal.json"
            if journal.exists():
                journal.unlink()
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("canonical publication transaction failed") from exc
        else:
            return destination

    def _write_current(
        self,
        attempt_id: str,
        digest: str,
        root_digest: str,
        files: dict[str, str],
        metadata: dict[str, object],
    ) -> None:
        raw_provenance = metadata.get("provenance", {})
        provenance: Provenance = {}
        if isinstance(raw_provenance, dict):
            parsed_provenance = cast("dict[str, object]", raw_provenance)
            provenance.update(parsed_provenance)
        current: CurrentManifest = {
            "dataset_id": self.root.name,
            "version": digest,
            "attempt_ids": [*self._current_attempts(), attempt_id],
            "files": files,
            "provenance": provenance,
            "root_digest": root_digest,
        }
        _atomic_json(self.root / "current.json", current)
        _atomic_json(self.root / "dataset_manifest.json", current)
        _atomic_json(self.root / "provenance.json", provenance)
        quality = {"attempt_id": attempt_id, "accepted": True, "mode": "human_gamepad"}
        quality_path = self.root / "quality.jsonl"
        with quality_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(quality, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_json(
            self.root / "splits.json", {"frozen": False, "episodes": [*self._current_attempts()]}
        )
        _atomic_json(
            self.root / "commit.json",
            {"attempt_id": attempt_id, "version": digest, "state": "committed"},
        )

    def _current_attempts(self) -> list[str]:
        current = self.root / "current.json"
        if not current.exists():
            return []
        value = json.loads(current.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return []
        loaded = cast("dict[str, object]", value)
        raw_ids = loaded.get("attempt_ids", [])
        if not isinstance(raw_ids, list):
            return []
        typed_ids = cast("list[object]", raw_ids)
        return [item for item in typed_ids if isinstance(item, str)]

    @staticmethod
    def _dataset_frame(frame: FrameRecord, task: str) -> dict[str, object]:
        return {
            **frame.observation,
            "action": np.asarray(frame.action, dtype=np.float32),
            "task": task,
        }

    @staticmethod
    def _feature_matches(value: object, dtype: str, shape: tuple[int, ...]) -> bool:
        if not isinstance(value, dict):
            return False
        feature = cast("dict[str, object]", value)
        raw_shape = feature.get("shape")
        return feature.get("dtype") == dtype and raw_shape in (shape, list(shape))

    @classmethod
    def _validate_features(cls, features: dict[str, object]) -> None:
        image_keys = {"observation.images.front", "observation.images.topdown"}
        present = image_keys & set(features)
        if len(present) != 1:
            raise ValueError("features must include exactly one policy image (front or topdown)")
        image_key = next(iter(present))
        expected = {image_key, "observation.state", "action"}
        if set(features) != expected:
            raise ValueError("features must be exactly image, state, and action")
        if not cls._feature_matches(features[image_key], "uint8", (96, 96, 3)):
            raise ValueError("policy image feature must be uint8 [96,96,3]")
        if not cls._feature_matches(features["observation.state"], "float32", (15,)):
            raise ValueError("state feature must be float32 [15]")
        if not cls._feature_matches(features["action"], "float32", (3,)):
            raise ValueError("action feature must be float32 [3]")

    @staticmethod
    def _require_human_metadata(metadata: dict[str, object]) -> None:
        if metadata.get("mode") != "human_gamepad" or metadata.get("physical_device") is not True:
            raise ValueError("only physical human_gamepad attempts may be canonical")
        if metadata.get("training_eligible") is not True:
            raise ValueError("attempt is not training eligible")

    @staticmethod
    def _lerobot_create(
        *, repo_id: str, root: Path, fps: int, features: dict[str, object]
    ) -> DatasetProtocol:
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            module = cast("_LeRobotModule", import_module("lerobot.datasets.lerobot_dataset"))
            create_method = module.LeRobotDataset.create
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("LeRobot 0.4.4 is required for canonical persistence") from exc
        return cast(DatasetProtocol, create_method(repo_id, fps, features, root=root))

    @staticmethod
    def reload_local(root: Path) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"local dataset root does not exist: {root}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            module = cast("_LeRobotModule", import_module("lerobot.datasets.lerobot_dataset"))
            dataset_type = module.LeRobotDataset
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("LeRobot 0.4.4 is required for local reload") from exc
        dataset_type(repo_id="local/so101-pusht-pilot", root=root, download_videos=False)

    _reload_local = reload_local
