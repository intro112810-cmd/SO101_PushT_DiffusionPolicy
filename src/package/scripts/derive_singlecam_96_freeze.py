#!/usr/bin/env python3
"""Create a single-camera 96x96 derived freeze from frozen_four_model_200ep.

Streams the zarr arrays chunk-by-chunk (never loading all 43k frames at once),
downscales cam_top 224x224 -> 96x96 with cv2 INTER_AREA, drops cam_side, and
writes a new frozen store whose manifest/digests are recomputed for the new
content. The source frozen store is never modified.

Usage (PUSHT_SINGLE_CAM=1 required):
    PUSHT_SINGLE_CAM=1 python scripts/derive_singlecam_96_freeze.py \
        --source datasets/frozen_four_model_200ep \
        --output datasets/frozen_four_model_200ep_s96 \
        --experiment-config config/experiment-200ep.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import cast

import cv2
import numpy as np
import zarr

from so101_pusht_benchmark.data.paper_view import (
    ARRAY_NAMES,
    PaperArray,
    canonical_digest,
    dtype_contract,
    root_provenance_digest,
)
from so101_pusht_benchmark.data.splits import (
    build_split_manifest,
    load_experiment_config,
)

_TARGET = 96
_EXPORTER_REVISION = "pusht_so100_native_v1"
_CONTRACT_SCHEMA = "pusht-so100-native-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_zarr_array(path: Path, rows: int) -> np.ndarray:
    """Read a full low-dim array (agent_pos/action/timestamp/...) into memory."""
    store = zarr.open(path, mode="r")
    return np.asarray(store[:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment-config", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get("PUSHT_SINGLE_CAM") != "1":
        raise SystemExit("PUSHT_SINGLE_CAM=1 is required for the derived single-cam freeze")

    source = args.source.resolve()
    output = args.output.resolve()
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    config = load_experiment_config(args.experiment_config)
    episode_ids = cast("list[str]", source_manifest["episode_ids"])
    source_digest = cast(str, source_manifest["canonical_digest"])
    if len(episode_ids) != config.target_episode_count:
        raise ValueError("derived freeze requires every source episode")

    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    staging_data = staging / "data"
    try:
        staging_data.mkdir(parents=True, exist_ok=True)
        # Low-dim arrays copied verbatim.
        for group in ("agent_pos", "action", "timestamp", "frame_index", "episode_id"):
            shutil.copytree(source / "data" / group, staging_data / group)
        if (source / "episode_ends").exists():
            shutil.copytree(source / "episode_ends", staging / "episode_ends")

        # cam_top: stream chunks, downscale.
        src_top = zarr.open(source / "data" / "cam_top", mode="r")
        rows = src_top.shape[0]
        dst_top = zarr.create(
            shape=(rows, _TARGET, _TARGET, 3),
            chunks=(1, _TARGET, _TARGET, 3),
            dtype="u1",
            store=staging_data / "cam_top",
            overwrite=False,
            compressor=None,
        )
        for start in range(0, rows, 256):
            end = min(start + 256, rows)
            block = src_top[start:end]
            resized = np.stack(
                [cv2.resize(frame, (_TARGET, _TARGET), interpolation=cv2.INTER_AREA) for frame in block]
            )
            dst_top[start:end] = resized
        print(f"cam_top resized: {rows} x {_TARGET}x{_TARGET}")

        # Restore group metadata that zarr.create may have skipped.
        (staging / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")
        (staging_data / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")

        # Recompute manifest for the new content.
        cam_top = np.asarray(zarr.open(staging_data / "cam_top", mode="r")[:])
        arrays = {
            "cam_top": cam_top,
            "agent_pos": _read_zarr_array(staging_data / "agent_pos", rows),
            "action": _read_zarr_array(staging_data / "action", rows),
            "timestamp": _read_zarr_array(staging_data / "timestamp", rows),
            "episode_id": _read_zarr_array(staging_data / "episode_id", rows),
            "frame_index": _read_zarr_array(staging_data / "frame_index", rows),
        }
        episode_ends = _read_zarr_array(staging / "episode_ends", 0).astype(np.int64)
        paper_arrays = {
            name: PaperArray(
                values=arrays[name],
                unit=cast("dict[str, str]", source_manifest["arrays"])[name]["unit"],
            )
            for name in ARRAY_NAMES
        }
        canonical = canonical_digest(paper_arrays, episode_ends, episode_ids)
        provenance = source_manifest["root_provenance"]
        root_digest = root_provenance_digest(provenance)
        records = {
            name: {
                "dtype": dtype_contract(arrays[name])[1],
                "shape": list(arrays[name].shape),
                "unit": paper_arrays[name].unit,
                "sha256": hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest(),
            }
            for name in ARRAY_NAMES
        }
        records["episode_ends"] = {
            "dtype": dtype_contract(episode_ends)[1],
            "shape": list(episode_ends.shape),
            "unit": "cumulative frames",
            "sha256": hashlib.sha256(np.ascontiguousarray(episode_ends).tobytes()).hexdigest(),
        }
        manifest = {
            "arrays": records,
            "canonical_digest": canonical,
            "contract_schema": _CONTRACT_SCHEMA,
            "episode_ids": episode_ids,
            "exporter_revision": _EXPORTER_REVISION,
            "fps": source_manifest["fps"],
            "root_digest": root_digest,
            "root_provenance": provenance,
            "runtime_lock_digest": source_manifest["runtime_lock_digest"],
            "splits": None,  # replaced below
            "training_eligible": True,
            "zarr_format": 2,
        }
        split = build_split_manifest(
            episode_ids,
            config,
            source_digest=canonical,
        ).to_dict()
        manifest["splits"] = split
        for name, value in (("splits.json", split), ("manifest.json", manifest)):
            temporary = staging / f".{name}.tmp"
            temporary.write_text(
                json.dumps(value, sort_keys=(name == "splits.json"), separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(staging / name)

        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), "canonical_digest": canonical}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
