from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import cast

from so101_pusht_benchmark.data.splits import (
    build_split_manifest,
    load_experiment_config,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment-config", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    config = load_experiment_config(args.experiment_config)
    episode_ids = cast("list[str]", source_manifest["episode_ids"])
    source_digest = cast(str, source_manifest["canonical_digest"])
    if len(episode_ids) != config.target_episode_count:
        raise ValueError("fast freeze requires every source episode")
    split = build_split_manifest(
        episode_ids,
        config,
        source_digest=source_digest,
    ).to_dict()
    staging = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    try:
        shutil.copytree(args.source, staging, copy_function=shutil.copy2)
        manifest = dict(source_manifest)
        manifest["splits"] = split
        manifest["training_eligible"] = True
        for name, value in (("splits.json", split), ("manifest.json", manifest)):
            temporary = staging / f".{name}.tmp"
            temporary.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(staging / name)
        if not (staging / "data/.zgroup").is_file() or not (
            staging / "episode_ends/.zarray"
        ).is_file():
            raise ValueError("copied native array tree is incomplete")
        staging.replace(args.output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"output": str(args.output), "split": split}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
