from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from so101_pusht_benchmark.training.paper_profiles import load_paper_profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--paper-view", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.paper_view / "manifest.json").read_text(encoding="utf-8"))
    split = json.loads((args.paper_view / "splits.json").read_text(encoding="utf-8"))
    profiles = load_paper_profiles(args.profiles)
    if args.output.exists():
        raise FileExistsError(args.output)
    staging = args.output.with_name(f".{args.output.name}.tmp")
    staging.mkdir(parents=True)
    try:
        for model, profile in profiles.models.items():
            for seed in profiles.training_seeds:
                run_id = f"{model}-seed-{seed}-200ep"
                value = {
                    "schema": "pusht-paper-faithful-run-v1",
                    "run_id": run_id,
                    "model": model,
                    "training_seed": seed,
                    "paper_view": str(args.paper_view.resolve()),
                    "dataset_digest": manifest["canonical_digest"],
                    "split_digest": split["digest"],
                    "budget": profile.budget,
                    "resolved_optimizer_updates": profile.resolved_optimizer_updates,
                    "parameters": profile.parameters,
                    "artifact_id": f"{model.replace('_', '-')}-200ep-seed-{seed}",
                    "output": f"models/200ep/{model}/seed-{seed}/full",
                    "systemd_unit": f"kihyun-pusht-{model.replace('_', '-')}-seed-{seed}-train.service",
                }
                encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
                (staging / f"{run_id}.json").write_text(encoded, encoding="utf-8")
        digest = hashlib.sha256(
            b"".join(path.read_bytes() for path in sorted(staging.glob("*.json")))
        ).hexdigest()
        (staging / "run-set.json").write_text(
            json.dumps(
                {
                    "schema": "pusht-paper-faithful-run-set-v1",
                    "runs": 12,
                    "digest": digest,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(args.output)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"runs": 12, "digest": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
