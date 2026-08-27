#!/usr/bin/env python3
"""Publish one synchronized 3-view DP-Transformer professor walkthrough."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import imageio.v2 as iio

sys.path.insert(0, str(Path(__file__).parent))
from generate_case_videos import CASE_SPECS
from generate_feedback_artifacts import capture_rollout, load_policy
from generate_layout_preview import three_view_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets = args.asset_dir.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    policy, identity = load_policy(
        args.artifact_root.resolve(), "local-dp_transformer-seed0", "dp_transformer"
    )
    output = assets / "2026-08-20_dp_transformer_professor_three_view_walkthrough_4x.mp4"
    with iio.get_writer(output, fps=40, macro_block_size=1) as writer:
        for _, label, seed, expected_success in CASE_SPECS:
            capture = capture_rollout(policy, identity, seed, label)
            if capture.success != expected_success:
                raise RuntimeError(f"seed {seed} no longer matches {label}")
            for frame in three_view_frames(capture):
                writer.append_data(frame)
    print(f"published synchronized 3-view walkthrough under {assets}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
