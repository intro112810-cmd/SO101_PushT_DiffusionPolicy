from __future__ import annotations

from itertools import accumulate
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .paper_view import (
    ARRAY_NAMES,
    LoadedPaperView,
    PaperArray,
    PaperViewError,
    PaperViewMetadata,
    canonical_digest,
    load_paper_view,
    require_sha256,
    root_provenance_digest,
    trusted_runtime_lock_digest,
    write_paper_view,
)


class MergeError(ValueError):
    """Raised when multiple native stores cannot form one safe lineage."""


def _episode_ranges(view: LoadedPaperView) -> dict[str, tuple[int, int]]:
    episode_ids = cast("list[str]", view.manifest["episode_ids"])
    starts = [0, *view.episode_ends.tolist()[:-1]]
    return dict(zip(episode_ids, zip(starts, view.episode_ends.tolist(), strict=True), strict=True))


def merge_paper_views(
    sources: list[Path],
    selections: list[list[str]],
    destination: Path,
    *,
    expected_episodes: int,
) -> Path:
    """Atomically merge explicit source episodes into one non-frozen native store."""
    if len(sources) != len(selections) or not sources:
        raise MergeError("sources and selections must be non-empty and aligned")
    if destination.exists():
        raise MergeError(f"output already exists: {destination}")
    if sum(len(items) for items in selections) != expected_episodes:
        raise MergeError(f"expected {expected_episodes} selected episodes")

    loaded_sources = [load_paper_view(source) for source in sources]
    rows_by_source: list[list[NDArray[np.int64]]] = []
    lengths: list[int] = []
    destination_ids: list[str] = []
    source_members: dict[str, str] = {}
    for source, view, selected in zip(sources, loaded_sources, selections, strict=True):
        if len(selected) != len(set(selected)):
            raise MergeError(f"duplicate episode selection: {source}")
        ranges = _episode_ranges(view)
        unknown = sorted(set(selected) - set(ranges))
        if unknown:
            raise MergeError(f"unknown episode selection: {unknown}")
        rows: list[NDArray[np.int64]] = []
        for episode_id in selected:
            start, end = ranges[episode_id]
            rows.append(np.arange(start, end, dtype=np.int64))
            lengths.append(end - start)
            destination_ids.append(f"{source.name}:{episode_id}")
        rows_by_source.append(rows)
        source_digest = require_sha256(
            view.manifest.get("canonical_digest"), "source canonical digest"
        )
        require_sha256(view.manifest.get("root_digest"), "source root digest")
        for episode_id in selected:
            source_members[f"{source.name}/episode-{episode_id}"] = source_digest

    arrays: dict[str, PaperArray] = {}
    for name in ARRAY_NAMES:
        parts: list[NDArray[np.generic]] = []
        for view, source_rows in zip(loaded_sources, rows_by_source, strict=True):
            if source_rows:
                indexes = np.concatenate(source_rows)
                parts.append(np.ascontiguousarray(view.arrays[name][indexes]))
        values = np.concatenate(parts)
        record = cast("dict[str, dict[str, object]]", loaded_sources[0].manifest["arrays"])
        arrays[name] = PaperArray(values, cast(str, record[name]["unit"]))

    total_frames = sum(lengths)
    episode_ordinals = np.empty(total_frames, dtype=np.int64)
    frame_ordinals = np.empty(total_frames, dtype=np.int64)
    cursor = 0
    for ordinal, length in enumerate(lengths):
        episode_ordinals[cursor : cursor + length] = ordinal
        frame_ordinals[cursor : cursor + length] = np.arange(length, dtype=np.int64)
        cursor += length
    arrays["episode_id"] = PaperArray(episode_ordinals, "episode ordinal")
    arrays["frame_index"] = PaperArray(frame_ordinals, "frame ordinal")
    arrays["timestamp"] = PaperArray(frame_ordinals.astype(np.float64) / 10, "seconds")
    episode_ends = np.asarray(list(accumulate(lengths)), dtype=np.int64)
    provenance: dict[str, object] = {
        "schema": "pusht-so100-root-provenance-v1",
        "source_members": source_members,
        "episodes": [
            {"episode_id": episode_id, "length": length}
            for episode_id, length in zip(destination_ids, lengths, strict=True)
        ],
    }
    metadata = PaperViewMetadata(
        canonical_digest(arrays, episode_ends, destination_ids),
        root_provenance_digest(provenance),
        provenance,
        destination_ids,
        {
            "frozen": False,
            "training_eligible": False,
            "reason": "split_manifest_not_frozen",
            "train": [],
            "validation": [],
            "test": [],
        },
        trusted_runtime_lock_digest(),
        False,
    )
    try:
        return write_paper_view(destination, arrays, episode_ends, metadata)
    except PaperViewError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise MergeError(f"merge failed: {exc}") from exc
