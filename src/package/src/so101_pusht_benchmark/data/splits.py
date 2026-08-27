"""Configurable deterministic episode selection and immutable split manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
from itertools import accumulate
import json
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import NDArray
import yaml

from .paper_view import (
    ARRAY_NAMES,
    PaperArray,
    PaperViewError,
    PaperViewMetadata,
    canonical_digest,
    require_sha256,
    root_provenance_digest,
    trusted_runtime_lock_digest,
    write_paper_view,
)
from .paper_view_reader import load_paper_view

_SPLIT_NAMES = ("train", "validation", "test")
_CONFIG_SCHEMA = "pusht-so100-experiment-v1"
_MANIFEST_SCHEMA = "pusht-so100-split-manifest-v1"
_DEFAULT_CONFIG = Path(__file__).parents[3] / "configs/experiment/pusht_so100_four_model_v1.yaml"


class SplitError(ValueError):
    """Raised when configuration, selection, or split membership is invalid."""


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Parsed decimal episode-budget configuration."""

    schema: str
    target_episode_count: int
    split_ratios: Mapping[str, Decimal]

    def __post_init__(self) -> None:
        """Defensively freeze caller-provided ratio mappings."""
        object.__setattr__(self, "split_ratios", MappingProxyType(dict(self.split_ratios)))


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Immutable exact episode selection shared by every model."""

    source_digest: str
    target_episode_count: int
    train_ratio: str
    validation_ratio: str
    test_ratio: str
    selected_episode_ids: tuple[str, ...]
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    sessions: tuple[tuple[str, str], ...] | None
    digest: str

    def members(self, name: str) -> tuple[str, ...]:
        if name not in _SPLIT_NAMES:
            raise SplitError(f"unknown split: {name}")
        return cast("tuple[str, ...]", getattr(self, name))

    def digest_payload(self) -> dict[str, object]:
        ratios = {
            "train": self.train_ratio,
            "validation": self.validation_ratio,
            "test": self.test_ratio,
        }
        splits = {name: list(self.members(name)) for name in _SPLIT_NAMES}
        return {
            "schema": _MANIFEST_SCHEMA,
            "source_digest": self.source_digest,
            "target_episode_count": self.target_episode_count,
            "split_ratios": ratios,
            "split_counts": {name: len(splits[name]) for name in _SPLIT_NAMES},
            "selected_episode_ids": list(self.selected_episode_ids),
            "splits": splits,
            "sessions": None if self.sessions is None else dict(self.sessions),
            "session_disjoint": self.sessions is not None,
            "frozen": True,
            "training_eligible": True,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.digest_payload(), "digest": self.digest}

    @classmethod
    def from_dict(cls, value: object) -> SplitManifest:
        if not isinstance(value, dict):
            raise SplitError("split manifest must be an object")
        raw = cast("dict[str, object]", value)
        expected = {
            "schema",
            "source_digest",
            "target_episode_count",
            "split_ratios",
            "split_counts",
            "selected_episode_ids",
            "splits",
            "sessions",
            "session_disjoint",
            "frozen",
            "training_eligible",
            "digest",
        }
        if set(raw) != expected or raw.get("schema") != _MANIFEST_SCHEMA:
            raise SplitError("split manifest fields are not exact")
        ratios_raw, counts_raw, splits_raw = (
            raw.get("split_ratios"),
            raw.get("split_counts"),
            raw.get("splits"),
        )
        if (
            not isinstance(ratios_raw, dict)
            or not isinstance(counts_raw, dict)
            or not isinstance(splits_raw, dict)
        ):
            raise SplitError("split manifest partitions are malformed")
        ratios = cast("dict[str, object]", ratios_raw)
        counts = cast("dict[str, object]", counts_raw)
        splits = cast("dict[str, object]", splits_raw)
        if (
            set(ratios) != set(_SPLIT_NAMES)
            or set(counts) != set(_SPLIT_NAMES)
            or set(splits) != set(_SPLIT_NAMES)
        ):
            raise SplitError("split manifest names are not exact")
        members: dict[str, tuple[str, ...]] = {}
        for name in _SPLIT_NAMES:
            values = splits[name]
            if not isinstance(values, list) or not all(
                isinstance(item, str) and item for item in cast("list[object]", values)
            ):
                raise SplitError("split manifest episode IDs are malformed")
            members[name] = tuple(cast("list[str]", values))
            if type(counts[name]) is not int or counts[name] != len(members[name]):
                raise SplitError("split manifest count mismatch")
        selected_raw = raw.get("selected_episode_ids")
        if not isinstance(selected_raw, list) or not all(
            isinstance(item, str) and item for item in cast("list[object]", selected_raw)
        ):
            raise SplitError("selected episode IDs are malformed")
        sessions_raw = raw.get("sessions")
        sessions: tuple[tuple[str, str], ...] | None
        if sessions_raw is None:
            sessions = None
        elif isinstance(sessions_raw, dict) and all(
            isinstance(key, str) and key and isinstance(item, str) and item
            for key, item in cast("dict[object, object]", sessions_raw).items()
        ):
            sessions = tuple(sorted(cast("dict[str, str]", sessions_raw).items()))
        else:
            raise SplitError("session metadata is malformed")
        target = raw.get("target_episode_count")
        digest = raw.get("digest")
        source = raw.get("source_digest")
        if type(target) is not int or not isinstance(digest, str) or not isinstance(source, str):
            raise SplitError("split manifest identity is malformed")
        if (
            raw.get("frozen") is not True
            or raw.get("training_eligible") is not True
            or raw.get("session_disjoint") is not (sessions is not None)
        ):
            raise SplitError("split manifest frozen state is invalid")
        for name in _SPLIT_NAMES:
            if not isinstance(ratios[name], str):
                raise SplitError("split manifest ratios are malformed")
        return cls(
            source,
            target,
            cast(str, ratios["train"]),
            cast(str, ratios["validation"]),
            cast(str, ratios["test"]),
            tuple(cast("list[str]", selected_raw)),
            members["train"],
            members["validation"],
            members["test"],
            sessions,
            digest,
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise SplitError(f"{label} must be a decimal number")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise SplitError(f"{label} must be a decimal number") from exc
    if not parsed.is_finite():
        raise SplitError(f"{label} must be finite")
    return parsed


def _ratio_text(value: Decimal) -> str:
    return format(value, "f")


def validate_experiment_config(config: ExperimentConfig) -> None:
    if config.schema != _CONFIG_SCHEMA:
        raise SplitError(f"experiment schema must be {_CONFIG_SCHEMA}")
    if type(config.target_episode_count) is not int or config.target_episode_count < 3:
        raise SplitError("target_episode_count must be at least 3")
    if set(config.split_ratios) != set(_SPLIT_NAMES):
        raise SplitError("split ratio names must be train, validation, test")
    ratios = [config.split_ratios[name] for name in _SPLIT_NAMES]
    if any(not value.is_finite() for value in ratios):
        raise SplitError("split ratios must be finite decimals")
    if config.split_ratios["train"] <= 0 or config.split_ratios["validation"] <= 0:
        raise SplitError("train and validation ratios must be positive")
    if config.split_ratios["test"] < 0:
        raise SplitError("test ratio must be non-negative")
    if sum(ratios, Decimal(0)) != Decimal(1):
        raise SplitError("split ratios must sum to 1 exactly as parsed decimals")


def load_experiment_config(path: Path = _DEFAULT_CONFIG) -> ExperimentConfig:
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SplitError(f"cannot load experiment config: {path}") from exc
    if not isinstance(value, dict):
        raise SplitError("experiment config must be an object")
    raw = cast("dict[str, object]", value)
    if set(raw) != {"schema", "target_episode_count", "split_ratios"}:
        raise SplitError("experiment config fields are not exact")
    ratios_raw = raw.get("split_ratios")
    if not isinstance(ratios_raw, dict) or set(cast("dict[object, object]", ratios_raw)) != set(
        _SPLIT_NAMES
    ):
        raise SplitError("split ratio names must be train, validation, test")
    ratios = cast("dict[str, object]", ratios_raw)
    target = raw.get("target_episode_count")
    if type(target) is not int:
        raise SplitError("target_episode_count must be an integer")
    config = ExperimentConfig(
        schema=cast(str, raw.get("schema")),
        target_episode_count=target,
        split_ratios={name: _decimal(ratios[name], f"{name} ratio") for name in _SPLIT_NAMES},
    )
    validate_experiment_config(config)
    allocate_split_counts(config)
    return config


def allocate_split_counts(config: ExperimentConfig) -> dict[str, int]:
    """Apply Hamilton's largest-remainder method with declared split-order ties."""
    validate_experiment_config(config)
    target = config.target_episode_count
    quotas = {name: config.split_ratios[name] * target for name in _SPLIT_NAMES}
    counts = {name: int(quotas[name]) for name in _SPLIT_NAMES}
    remaining = target - sum(counts.values())
    order = sorted(
        _SPLIT_NAMES,
        key=lambda name: (-(quotas[name] - counts[name]), _SPLIT_NAMES.index(name)),
    )
    for name in order[:remaining]:
        counts[name] += 1
    if sum(counts.values()) != target:
        raise SplitError("largest-remainder allocation total mismatch")
    if any(
        counts[name] < 1
        for name in _SPLIT_NAMES
        if config.split_ratios[name] > 0
    ):
        raise SplitError("non-empty positive-ratio allocation impossible")
    if any(
        counts[name] != 0
        for name in _SPLIT_NAMES
        if config.split_ratios[name] == 0
    ):
        raise SplitError("zero-ratio split allocation must be empty")
    return counts


def _assign_sessions(
    selected: tuple[str, ...], sessions: Mapping[str, str], counts: Mapping[str, int]
) -> dict[str, list[str]]:
    if set(selected) - set(sessions):
        raise SplitError("session metadata must cover every selected episode")
    by_session: dict[str, list[str]] = {}
    for episode_id in selected:
        session = sessions[episode_id]
        if not session:
            raise SplitError("session metadata must contain non-empty session IDs")
        by_session.setdefault(session, []).append(episode_id)
    groups = sorted(by_session.items(), key=lambda item: (-len(item[1]), item[0]))
    initial = tuple(counts[name] for name in _SPLIT_NAMES)
    failed: set[tuple[int, tuple[int, int, int]]] = set()

    def search(index: int, remaining: tuple[int, int, int]) -> tuple[int, ...] | None:
        state = (index, remaining)
        if state in failed:
            return None
        if index == len(groups):
            return () if remaining == (0, 0, 0) else None
        size = len(groups[index][1])
        tried_capacity: set[int] = set()
        for bucket, capacity in enumerate(remaining):
            if capacity < size or capacity in tried_capacity:
                continue
            tried_capacity.add(capacity)
            updated = list(remaining)
            updated[bucket] -= size
            suffix = search(index + 1, cast("tuple[int, int, int]", tuple(updated)))
            if suffix is not None:
                return (bucket, *suffix)
        failed.add(state)
        return None

    assignment = search(0, cast("tuple[int, int, int]", initial))
    if assignment is None:
        raise SplitError("session-disjoint allocation impossible for configured split counts")
    result: dict[str, list[str]] = {name: [] for name in _SPLIT_NAMES}
    for (_, members), bucket in zip(groups, assignment, strict=True):
        result[_SPLIT_NAMES[bucket]].extend(members)
    selected_order = {episode_id: index for index, episode_id in enumerate(selected)}
    for name in _SPLIT_NAMES:
        result[name].sort(key=selected_order.__getitem__)
    return result


def _manifest_digest(manifest: SplitManifest) -> str:
    return hashlib.sha256(_canonical_json(manifest.digest_payload())).hexdigest()


def build_split_manifest(
    eligible_episode_ids: list[str] | tuple[str, ...],
    config: ExperimentConfig,
    *,
    source_digest: str,
    sessions: Mapping[str, str] | None = None,
) -> SplitManifest:
    counts = allocate_split_counts(config)
    require_sha256(source_digest, "split source digest")
    if any(not item for item in eligible_episode_ids):
        raise SplitError("eligible episode IDs must be non-empty strings")
    if len(set(eligible_episode_ids)) != len(eligible_episode_ids):
        raise SplitError("duplicate eligible episode ID")
    accepted = len(eligible_episode_ids)
    target = config.target_episode_count
    if accepted < target:
        missing = target - accepted
        raise SplitError(f"accepted episodes {accepted}/{target}; collect {missing} more")
    selected = tuple(sorted(eligible_episode_ids)[:target])
    if sessions is None:
        split: dict[str, list[str]] = {}
        cursor = 0
        for name in _SPLIT_NAMES:
            split[name] = list(selected[cursor : cursor + counts[name]])
            cursor += counts[name]
        session_pairs = None
    else:
        split = _assign_sessions(selected, sessions, counts)
        session_pairs = tuple((episode_id, sessions[episode_id]) for episode_id in selected)
    ratios = [_ratio_text(config.split_ratios[name]) for name in _SPLIT_NAMES]
    provisional = SplitManifest(
        source_digest,
        target,
        ratios[0],
        ratios[1],
        ratios[2],
        selected,
        tuple(split["train"]),
        tuple(split["validation"]),
        tuple(split["test"]),
        session_pairs,
        "0" * 64,
    )
    manifest = SplitManifest(
        source_digest,
        target,
        ratios[0],
        ratios[1],
        ratios[2],
        selected,
        tuple(split["train"]),
        tuple(split["validation"]),
        tuple(split["test"]),
        session_pairs,
        _manifest_digest(provisional),
    )
    validate_split_manifest(manifest, eligible_episode_ids, config, source_digest=source_digest)
    return manifest


def validate_split_manifest(
    manifest: SplitManifest,
    eligible_episode_ids: list[str] | tuple[str, ...],
    config: ExperimentConfig,
    *,
    source_digest: str,
) -> None:
    counts = allocate_split_counts(config)
    require_sha256(source_digest, "split source digest")
    if (
        manifest.source_digest != source_digest
        or manifest.target_episode_count != config.target_episode_count
    ):
        raise SplitError("split manifest source or target mismatch")
    expected_ratios = tuple(_ratio_text(config.split_ratios[name]) for name in _SPLIT_NAMES)
    if (manifest.train_ratio, manifest.validation_ratio, manifest.test_ratio) != expected_ratios:
        raise SplitError("split manifest ratio mismatch")
    if (
        len(manifest.selected_episode_ids) != config.target_episode_count
        or tuple(sorted(eligible_episode_ids)[: config.target_episode_count])
        != manifest.selected_episode_ids
    ):
        raise SplitError("selected episodes are not the deterministic exact budget")
    flattened: list[str] = []
    for name in _SPLIT_NAMES:
        members = manifest.members(name)
        if len(members) != counts[name] or (counts[name] > 0 and not members):
            raise SplitError(f"{name} split count mismatch")
        flattened.extend(members)
    if len(flattened) != len(set(flattened)):
        raise SplitError("split episode overlap is forbidden")
    if set(flattened) != set(manifest.selected_episode_ids):
        raise SplitError("splits are not an exact selected-episode partition")
    if manifest.sessions is not None:
        session_map = dict(manifest.sessions)
        if set(session_map) != set(manifest.selected_episode_ids):
            raise SplitError("session metadata must cover every selected episode")
        split_sessions = [
            {session_map[item] for item in manifest.members(name)} for name in _SPLIT_NAMES
        ]
        if any(
            split_sessions[left] & split_sessions[right]
            for left in range(len(_SPLIT_NAMES))
            for right in range(left + 1, len(_SPLIT_NAMES))
        ):
            raise SplitError("session overlap across splits is forbidden")
    if manifest.digest != _manifest_digest(manifest):
        raise SplitError("split manifest digest mismatch")


def build_splits(
    episodes: Mapping[str, str] | list[str] | tuple[str, ...],
    config: ExperimentConfig | None = None,
    *,
    target_episode_count: int | None = None,
    ratios: Mapping[str, Decimal | str | float | int] | None = None,
) -> dict[str, list[str]]:
    """Compatibility entrypoint backed only by the versioned configurable contract."""
    if config is None:
        config = load_experiment_config()
    if target_episode_count is not None or ratios is not None:
        config = ExperimentConfig(
            config.schema,
            config.target_episode_count if target_episode_count is None else target_episode_count,
            config.split_ratios
            if ratios is None
            else {name: _decimal(ratios[name], f"{name} ratio") for name in _SPLIT_NAMES},
        )
    ids = list(episodes)
    sessions = episodes if isinstance(episodes, Mapping) else None
    manifest = build_split_manifest(ids, config, source_digest="0" * 64, sessions=sessions)
    return {name: list(manifest.members(name)) for name in _SPLIT_NAMES}


def validate_splits(
    split: Mapping[str, list[str]],
    episodes: Mapping[str, str] | list[str] | tuple[str, ...],
    config: ExperimentConfig | None = None,
) -> None:
    if config is None:
        config = load_experiment_config()
    ids = list(episodes)
    sessions = episodes if isinstance(episodes, Mapping) else None
    expected = build_split_manifest(ids, config, source_digest="0" * 64, sessions=sessions)
    if set(split) != set(_SPLIT_NAMES) or any(
        tuple(split[name]) != expected.members(name) for name in _SPLIT_NAMES
    ):
        raise SplitError("split membership differs from deterministic manifest")


def load_and_validate(
    path: Path,
    episodes: Mapping[str, str] | list[str] | tuple[str, ...],
    config: ExperimentConfig | None = None,
) -> dict[str, list[str]]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitError("split file is not valid JSON") from exc
    if not isinstance(value, dict):
        raise SplitError("split file must be an object")
    raw = cast("dict[str, object]", value)
    if set(raw) != set(_SPLIT_NAMES):
        raise SplitError("split names must be train, validation, test")
    split: dict[str, list[str]] = {}
    for name in _SPLIT_NAMES:
        members = raw[name]
        if not isinstance(members, list) or not all(
            isinstance(member, str) for member in cast("list[object]", members)
        ):
            raise SplitError("split members must be episode ID lists")
        split[name] = cast("list[str]", members)
    validate_splits(split, episodes, config)
    return split


def _selected_arrays(
    source: Path, manifest: SplitManifest
) -> tuple[dict[str, PaperArray], NDArray[np.int64], dict[str, object]]:
    loaded = load_paper_view(source)
    source_ids = cast("list[str]", loaded.manifest["episode_ids"])
    starts = [0, *loaded.episode_ends.tolist()[:-1]]
    ranges = dict(
        zip(source_ids, zip(starts, loaded.episode_ends.tolist(), strict=True), strict=True)
    )
    rows: list[NDArray[np.int64]] = []
    lengths: list[int] = []
    for episode_id in manifest.selected_episode_ids:
        start, end = ranges[episode_id]
        rows.append(np.arange(start, end, dtype=np.int64))
        lengths.append(end - start)
    selected_rows = np.asarray([value for row in rows for value in row.tolist()], dtype=np.int64)
    records = cast("dict[str, dict[str, object]]", loaded.manifest["arrays"])
    arrays: dict[str, PaperArray] = {}
    for name in ARRAY_NAMES:
        selected_values = np.ascontiguousarray(loaded.arrays[name][selected_rows])
        arrays[name] = PaperArray(selected_values, cast(str, records[name]["unit"]))
    ordinal_values = np.empty(sum(lengths), dtype=np.int64)
    cursor = 0
    for ordinal, length in enumerate(lengths):
        ordinal_values[cursor : cursor + length] = ordinal
        cursor += length
    arrays["episode_id"] = PaperArray(ordinal_values, "episode ordinal")
    ends = np.asarray(list(accumulate(lengths)), dtype=np.int64)
    source_provenance = cast("dict[str, object]", loaded.manifest["root_provenance"])
    provenance: dict[str, object] = {
        "schema": "pusht-so100-root-provenance-v1",
        "source_members": source_provenance["source_members"],
        "episodes": [
            {"episode_id": episode_id, "length": length}
            for episode_id, length in zip(manifest.selected_episode_ids, lengths, strict=True)
        ],
    }
    return arrays, ends, provenance


def freeze_training_view(
    source: Path,
    destination: Path,
    config: ExperimentConfig,
    *,
    sessions: Mapping[str, str] | None = None,
) -> tuple[Path, SplitManifest]:
    """Select an exact budget and atomically publish one immutable training view."""
    loaded = load_paper_view(source)
    source_digest = require_sha256(loaded.manifest.get("canonical_digest"), "source digest")
    episode_ids = cast("list[str]", loaded.manifest["episode_ids"])
    manifest = build_split_manifest(
        episode_ids, config, source_digest=source_digest, sessions=sessions
    )
    if destination.exists():
        existing = load_paper_view(destination)
        existing_manifest = SplitManifest.from_dict(existing.splits)
        validate_split_manifest(existing_manifest, episode_ids, config, source_digest=source_digest)
        if existing_manifest != manifest:
            raise SplitError("immutable split manifest differs from requested inputs")
        return destination, existing_manifest
    arrays, ends, provenance = _selected_arrays(source, manifest)
    selected_ids = list(manifest.selected_episode_ids)
    metadata = PaperViewMetadata(
        canonical_digest(arrays, ends, selected_ids),
        root_provenance_digest(provenance),
        provenance,
        selected_ids,
        manifest.to_dict(),
        trusted_runtime_lock_digest(),
        True,
    )
    try:
        return write_paper_view(destination, arrays, ends, metadata), manifest
    except PaperViewError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise SplitError(f"split freeze failed: {exc}") from exc
