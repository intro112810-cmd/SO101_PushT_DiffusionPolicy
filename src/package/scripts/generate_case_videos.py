#!/usr/bin/env python3
"""Publish labeled DP-Transformer success and failure case videos."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from generate_feedback_artifacts import capture_rollout, load_policy, write_video


CASE_SPECS = (
    ("success_long", "SUCCESS: sustained push", 100053, True),
    ("success_precise", "SUCCESS: precise finish", 100033, True),
    ("failure_yaw", "FAILURE: orientation mismatch", 100008, False),
    ("failure_position", "FAILURE: position mismatch", 100060, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.artifact_root.resolve()
    assets = args.asset_dir.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    policy, identity = load_policy(root, "local-dp_transformer-seed0", "dp_transformer")
    for name, label, seed, expected_success in CASE_SPECS:
        capture = capture_rollout(policy, identity, seed, label)
        if capture.success != expected_success:
            raise RuntimeError(f"seed {seed} no longer matches {label}")
        write_video(
            (capture,),
            assets / f"2026-08-20_dp_transformer_case_{name}_4x.mp4",
        )
    print(f"published {len(CASE_SPECS)} case videos under {assets}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
