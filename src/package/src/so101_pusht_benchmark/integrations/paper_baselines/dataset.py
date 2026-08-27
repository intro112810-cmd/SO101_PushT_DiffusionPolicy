"""Shared immutable native dataset for unchanged Stanford image policies.

Upstream symbols: BaseImageDataset and LinearNormalizer at Stanford commit
5ba07ac6661db573af695b419a7947ecb704690f. This adapter owns storage, frozen
split selection, episode-bounded windows, and feature translation only.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
import copy
from decimal import Decimal
import os
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

from ...data.paper_view import LoadedPaperView, PaperViewError
from ...data.paper_view_reader import load_paper_view, validate_training_view
from ...data.splits import (
    ExperimentConfig,
    SplitError,
    SplitManifest,
    validate_split_manifest,
)

_SINGLE_CAM = os.environ.get("PUSHT_SINGLE_CAM") == "1"

if _SINGLE_CAM:
    _ARRAY_UNITS = {
        "cam_top": "rgb intensity",
        "agent_pos": "radians",
        "action": "absolute normalized mocap XY",
        "timestamp": "seconds",
        "episode_id": "episode ordinal",
        "frame_index": "frame ordinal",
        "episode_ends": "cumulative frames",
    }
    _POLICY_ARRAY_KEYS = (
        "cam_top",
        "agent_pos",
        "action",
        "timestamp",
        "episode_id",
        "frame_index",
    )
    _POLICY_OBSERVATION_KEYS = ("cam_top", "agent_pos")
    _POLICY_NORMALIZER_KEYS = ("cam_top", "agent_pos", "action")
else:
    _ARRAY_UNITS = {
        "cam_top": "rgb intensity",
        "cam_side": "rgb intensity",
        "agent_pos": "radians",
        "action": "absolute normalized mocap XY",
        "timestamp": "seconds",
        "episode_id": "episode ordinal",
        "frame_index": "frame ordinal",
        "episode_ends": "cumulative frames",
    }
    _POLICY_ARRAY_KEYS = (
        "cam_top",
        "cam_side",
        "agent_pos",
        "action",
        "timestamp",
        "episode_id",
        "frame_index",
    )
    _POLICY_OBSERVATION_KEYS = ("cam_top", "cam_side", "agent_pos")
    _POLICY_NORMALIZER_KEYS = ("cam_top", "cam_side", "agent_pos", "action")

_from_numpy = cast("Callable[[NDArray[np.generic]], torch.Tensor]", torch.from_numpy)
_REPEAT_TO_SAMPLES = ContextVar[int | None]("paper_baseline_repeat_to_samples", default=None)


class DatasetNamespaceError(PaperViewError):
    """The frozen view cannot satisfy the exact Stanford policy namespace."""


class _ParameterOwner(Protocol):
    params_dict: torch.nn.ParameterDict


@contextmanager
def repeat_training_samples(samples: int) -> Generator[None]:
    """Repeat only a train dataset to one exact sample budget within this context."""
    if samples < 1:
        raise ValueError("repeat training samples must be positive")
    token = _REPEAT_TO_SAMPLES.set(samples)
    try:
        yield
    finally:
        _REPEAT_TO_SAMPLES.reset(token)


def validate_policy_arrays(
    arrays: dict[str, NDArray[np.generic]], episode_ends: NDArray[np.int64]
) -> None:
    """Reject storage that cannot produce exact dual-camera policy tensors."""
    if tuple(arrays) != _POLICY_ARRAY_KEYS:
        raise DatasetNamespaceError("native policy array keys/order mismatch")
    if episode_ends.ndim != 1 or episode_ends.dtype != np.dtype(np.int64) or not len(episode_ends):
        raise DatasetNamespaceError("episode_ends must be non-empty int64[episodes]")
    rows = int(episode_ends[-1])
    if _SINGLE_CAM:
        expected = {
            "cam_top": ((rows, 96, 96, 3), np.dtype(np.uint8)),
            "agent_pos": ((rows, 5), np.dtype(np.float32)),
            "action": ((rows, 2), np.dtype(np.float32)),
            "timestamp": ((rows,), np.dtype(np.float64)),
            "episode_id": ((rows,), np.dtype(np.int64)),
            "frame_index": ((rows,), np.dtype(np.int64)),
        }
    else:
        expected = {
            "cam_top": ((rows, 224, 224, 3), np.dtype(np.uint8)),
            "cam_side": ((rows, 224, 224, 3), np.dtype(np.uint8)),
            "agent_pos": ((rows, 5), np.dtype(np.float32)),
            "action": ((rows, 2), np.dtype(np.float32)),
            "timestamp": ((rows,), np.dtype(np.float64)),
            "episode_id": ((rows,), np.dtype(np.int64)),
            "frame_index": ((rows,), np.dtype(np.int64)),
        }
    for key, (shape, dtype) in expected.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != dtype:
            raise DatasetNamespaceError(f"{key} must have shape {shape} and dtype {dtype.name}")


def validate_policy_normalizer(normalizer: LinearNormalizer) -> None:
    """Reject normalizer state that does not match the shared policy namespace."""
    parameters = cast("_ParameterOwner", normalizer).params_dict
    if tuple(parameters) != _POLICY_NORMALIZER_KEYS:
        raise DatasetNamespaceError("normalizer keys/order mismatch")
    if _SINGLE_CAM:
        expected_width = {"cam_top": 1, "agent_pos": 5, "action": 2}
    else:
        expected_width = {"cam_top": 1, "cam_side": 1, "agent_pos": 5, "action": 2}
    for key, width in expected_width.items():
        field = cast("_ParameterOwner", normalizer[key])
        for parameter in ("offset", "scale"):
            value = field.params_dict[parameter]
            if tuple(value.shape) != (width,) or value.dtype is not torch.float32:
                raise DatasetNamespaceError(f"normalizer {key} {parameter} shape/dtype mismatch")


def validate_policy_sample(sample: object, horizon: int) -> None:
    """Reject a sampled window before it can reach an upstream policy."""
    if not isinstance(sample, dict):
        raise DatasetNamespaceError("sample keys/order must be obs, action")
    root = cast("dict[str, object]", sample)
    if tuple(root) != ("obs", "action"):
        raise DatasetNamespaceError("sample keys/order must be obs, action")
    observation = root["obs"]
    action = root["action"]
    if not isinstance(observation, dict):
        raise DatasetNamespaceError("observation keys/order mismatch")
    observations = cast("dict[str, object]", observation)
    if tuple(observations) != _POLICY_OBSERVATION_KEYS:
        raise DatasetNamespaceError("observation keys/order mismatch")
    if _SINGLE_CAM:
        expected = {
            "cam_top": (horizon, 3, 96, 96),
            "agent_pos": (horizon, 5),
        }
    else:
        expected = {
            "cam_top": (horizon, 3, 224, 224),
            "cam_side": (horizon, 3, 224, 224),
            "agent_pos": (horizon, 5),
        }
    for key, shape in expected.items():
        tensor = observations[key]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise DatasetNamespaceError(f"{key} tensor must have shape {shape}")
        if tensor.dtype is not torch.float32:
            raise DatasetNamespaceError(f"{key} tensor must have dtype float32")
    if not isinstance(action, torch.Tensor) or tuple(action.shape) != (horizon, 2):
        raise DatasetNamespaceError(f"action tensor must have shape ({horizon}, 2)")
    if action.dtype is not torch.float32:
        raise DatasetNamespaceError("action tensor must have dtype float32")


class PaperBaselineDataset(BaseImageDataset):
    """Expose one frozen native view through Stanford's dataset contract."""

    def __init__(
        self,
        zarr_path: str | Path,
        horizon: int,
        pad_before: int = 0,
        pad_after: int = 0,
        split: str = "train",
    ) -> None:
        super().__init__()
        if horizon < 1 or not 0 <= pad_before < horizon or not 0 <= pad_after < horizon:
            raise ValueError("invalid horizon or padding")
        self.path = Path(zarr_path)
        self._synthetic_probe = split == "synthetic_probe"
        try:
            self.view = (
                load_paper_view(self.path)
                if self._synthetic_probe
                else validate_training_view(self.path)
            )
        except PaperViewError as exc:
            raise DatasetNamespaceError(f"native policy dataset contract failed: {exc}") from exc
        validate_policy_arrays(self.view.arrays, self.view.episode_ends)
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.repeat_to_samples = _REPEAT_TO_SAMPLES.get() if split == "train" else None
        self._manifest = self._validate_contract(self.view, self._synthetic_probe)
        self._episode_ranges = self._ranges(self.view)
        self._train_rows = self._rows("train")
        self.select_split(split)

    @staticmethod
    def _validate_contract(view: LoadedPaperView, synthetic_probe: bool) -> SplitManifest | None:
        records_raw = view.manifest.get("arrays")
        episode_ids_raw = view.manifest.get("episode_ids")
        if not isinstance(records_raw, dict) or not isinstance(episode_ids_raw, list):
            raise PaperViewError("native dataset manifest is malformed")
        records = cast("dict[str, object]", records_raw)
        if set(records) != set(_ARRAY_UNITS):
            raise PaperViewError("native dataset array keys are not exact")
        for key, unit in _ARRAY_UNITS.items():
            record = records[key]
            if (
                not isinstance(record, dict)
                or cast("dict[str, object]", record).get("unit") != unit
            ):
                raise PaperViewError(f"native dataset unit mismatch: {key}")
        episode_ids = cast("list[object]", episode_ids_raw)
        if not all(isinstance(item, str) and item for item in episode_ids) or len(
            set(cast("list[str]", episode_ids))
        ) != len(episode_ids):
            raise PaperViewError("episode IDs must be unique non-empty strings")
        if synthetic_probe:
            return None
        try:
            manifest = SplitManifest.from_dict(view.splits)
            config = ExperimentConfig(
                schema="pusht-so100-experiment-v1",
                target_episode_count=manifest.target_episode_count,
                split_ratios={
                    "train": Decimal(manifest.train_ratio),
                    "validation": Decimal(manifest.validation_ratio),
                    "test": Decimal(manifest.test_ratio),
                },
            )
            validate_split_manifest(
                manifest,
                cast("list[str]", episode_ids),
                config,
                source_digest=manifest.source_digest,
            )
        except (SplitError, ArithmeticError) as exc:
            raise PaperViewError(f"invalid frozen split manifest: {exc}") from exc
        ordinals = cast("NDArray[np.int64]", view.arrays["episode_id"])
        frames = cast("NDArray[np.int64]", view.arrays["frame_index"])
        starts = [0, *view.episode_ends[:-1].tolist()]
        for ordinal, (start, end) in enumerate(
            zip(starts, view.episode_ends.tolist(), strict=True)
        ):
            if not bool(np.all(ordinals[start:end] == ordinal)) or not np.array_equal(
                frames[start:end], np.arange(end - start, dtype=np.int64)
            ):
                raise PaperViewError("episode ordinal or frame index mismatch")
        return manifest

    @staticmethod
    def _ranges(view: LoadedPaperView) -> dict[str, tuple[int, int]]:
        ids = cast("list[str]", view.manifest["episode_ids"])
        starts = [0, *view.episode_ends[:-1].tolist()]
        return dict(zip(ids, zip(starts, view.episode_ends.tolist(), strict=True), strict=True))

    def _members(self, split: str) -> tuple[str, ...]:
        if self._synthetic_probe:
            if split not in {"synthetic_probe", "train", "validation", "test"}:
                raise SplitError(f"unknown split: {split}")
            return tuple(cast("list[str]", self.view.manifest["episode_ids"]))
        if self._manifest is None:
            raise PaperViewError("frozen split manifest is unavailable")
        return self._manifest.members(split)

    def _rows(self, split: str) -> NDArray[np.int64]:
        chunks = [
            np.arange(*self._episode_ranges[item], dtype=np.int64) for item in self._members(split)
        ]
        return np.concatenate(chunks)

    def select_split(self, split: str) -> None:
        """Select one manifest partition and preserve its exact episode/window order."""
        self.split = split
        self.episode_ids = self._members(split)
        windows: list[tuple[int, int, int]] = []
        ordered: list[tuple[str, int]] = []
        for episode_id in self.episode_ids:
            start, end = self._episode_ranges[episode_id]
            length = end - start
            offsets = range(-self.pad_before, length - self.horizon + self.pad_after + 1)
            for offset in offsets:
                windows.append((start, end, offset))
                ordered.append((episode_id, offset))
        if not windows:
            raise PaperViewError(f"no sequence windows for {split}")
        self._windows = tuple(windows)
        self.ordered_windows = tuple(ordered)

    def get_validation_dataset(self) -> PaperBaselineDataset:
        result = copy.copy(self)
        result.repeat_to_samples = None
        result.select_split("validation")
        return result

    def get_test_dataset(self) -> PaperBaselineDataset:
        result = copy.copy(self)
        result.repeat_to_samples = None
        result.select_split("test")
        return result

    def get_normalizer(self, mode: str = "limits") -> LinearNormalizer:
        """Fit numeric fields from selected train rows and fixed ranges for cameras."""
        normalizer = LinearNormalizer()
        normalizer["cam_top"] = get_image_range_normalizer()
        if not _SINGLE_CAM:
            normalizer["cam_side"] = get_image_range_normalizer()
        actions = cast("NDArray[np.float32]", self.view.arrays["action"])
        states = cast("NDArray[np.float32]", self.view.arrays["agent_pos"])
        normalizer.fit(
            {"agent_pos": states[self._train_rows], "action": actions[self._train_rows]},
            last_n_dims=1,
            mode=mode,
        )
        validate_policy_normalizer(normalizer)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        actions = cast("NDArray[np.float32]", self.view.arrays["action"])
        return _from_numpy(actions[self._train_rows].copy())

    def __len__(self) -> int:
        """Return the exact number of ordered windows in the selected split."""
        if self.repeat_to_samples is not None:
            return self.repeat_to_samples
        return len(self._windows)

    def sample_window(self, index: int) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        """Return one exact ordered episode-bounded window."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self._sample(self._windows[index % len(self._windows)])

    def _sample(
        self, window: tuple[int, int, int]
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        episode_start, episode_end, offset = window
        relative = np.clip(
            np.arange(offset, offset + self.horizon), 0, episode_end - episode_start - 1
        )
        indices = relative + episode_start
        observations: dict[str, torch.Tensor] = {}
        cameras = ("cam_top",) if _SINGLE_CAM else ("cam_top", "cam_side")
        for camera in cameras:
            images = cast("NDArray[np.uint8]", self.view.arrays[camera])
            image = np.moveaxis(images[indices], -1, 1).astype(np.float32) / np.float32(255)
            observations[camera] = _from_numpy(image)
        states = cast("NDArray[np.float32]", self.view.arrays["agent_pos"])
        actions = cast("NDArray[np.float32]", self.view.arrays["action"])
        observations["agent_pos"] = _from_numpy(states[indices].copy())
        sample: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {
            "obs": observations,
            "action": _from_numpy(actions[indices].copy()),
        }
        validate_policy_sample(sample, self.horizon)
        return sample

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        """Return the exact ordered window without virtual-index remapping."""
        return self.sample_window(index)
