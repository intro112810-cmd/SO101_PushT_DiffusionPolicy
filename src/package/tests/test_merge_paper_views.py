from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from so101_pusht_benchmark.data.merge import MergeError, merge_paper_views
from so101_pusht_benchmark.data.paper_view import (
    PaperArray,
    PaperViewMetadata,
    canonical_digest,
    load_paper_view,
    root_provenance_digest,
    trusted_runtime_lock_digest,
    write_paper_view,
)


def _view(root: Path, name: str, lengths: list[int]) -> Path:
    total = sum(lengths)
    ends = np.cumsum(lengths, dtype=np.int64)
    ids = [str(index) for index in range(len(lengths))]
    ordinal = np.concatenate(
        [np.full(length, index, dtype=np.int64) for index, length in enumerate(lengths)]
    )
    frame = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
    arrays = {
        "cam_top": PaperArray(np.zeros((total, 224, 224, 3), dtype=np.uint8), "rgb intensity"),
        "cam_side": PaperArray(np.ones((total, 224, 224, 3), dtype=np.uint8), "rgb intensity"),
        "agent_pos": PaperArray(np.zeros((total, 5), dtype=np.float32), "radians"),
        "action": PaperArray(np.zeros((total, 2), dtype=np.float32), "absolute normalized mocap XY"),
        "timestamp": PaperArray(frame.astype(np.float64) / 10, "seconds"),
        "episode_id": PaperArray(ordinal, "episode ordinal"),
        "frame_index": PaperArray(frame, "frame ordinal"),
    }
    provenance: dict[str, object] = {
        "schema": "pusht-so100-root-provenance-v1",
        "source_members": {f"{name}/source": "a" * 64},
        "episodes": [
            {"episode_id": episode_id, "length": length}
            for episode_id, length in zip(ids, lengths, strict=True)
        ],
    }
    output = root / name
    write_paper_view(
        output,
        arrays,
        ends,
        PaperViewMetadata(
            canonical_digest(arrays, ends, ids),
            root_provenance_digest(provenance),
            provenance,
            ids,
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
        ),
    )
    return output


def test_merge_exact_episode_selection_and_provenance(canonical_test_root: Path) -> None:
    first = _view(canonical_test_root, "first", [2, 3])
    second = _view(canonical_test_root, "second", [4, 5, 6])
    output = canonical_test_root / "merged"

    merge_paper_views(
        [first, second],
        [["0", "1"], ["0", "2"]],
        output,
        expected_episodes=4,
    )

    loaded = load_paper_view(output)
    assert loaded.manifest["episode_ids"] == ["first:0", "first:1", "second:0", "second:2"]
    assert loaded.episode_ends.tolist() == [2, 5, 9, 15]
    assert loaded.arrays["episode_id"].tolist() == [0, 0, 1, 1, 1, *([2] * 4), *([3] * 6)]
    assert loaded.manifest["training_eligible"] is False
    provenance = cast("dict[str, Any]", loaded.manifest["root_provenance"])
    assert provenance["source_members"] == {
        "first/episode-0": loaded_sources_digest(first),
        "first/episode-1": loaded_sources_digest(first),
        "second/episode-0": loaded_sources_digest(second),
        "second/episode-2": loaded_sources_digest(second),
    }


def loaded_sources_digest(path: Path) -> str:
    return cast(str, load_paper_view(path).manifest["canonical_digest"])


@pytest.mark.parametrize(
    ("selections", "expected"),
    [
        ([["0", "0"], ["0", "1"]], "duplicate episode selection"),
        ([["0"], ["0", "1"]], "expected 4 selected episodes"),
        ([["9", "1"], ["0", "1"]], "unknown episode selection"),
    ],
)
def test_merge_rejects_unsafe_selection_without_output(
    canonical_test_root: Path, selections: list[list[str]], expected: str
) -> None:
    first = _view(canonical_test_root, "first", [2, 3])
    second = _view(canonical_test_root, "second", [4, 5])
    output = canonical_test_root / "merged"

    with pytest.raises(MergeError, match=expected):
        merge_paper_views([first, second], selections, output, expected_episodes=4)

    assert not output.exists()
