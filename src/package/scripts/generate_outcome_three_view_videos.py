#!/usr/bin/env python3
"""Publish approximately two-minute success and failure 3-view rollouts."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
from typing import Final

import imageio.v2 as iio

sys.path.insert(0, str(Path(__file__).parent))
from generate_feedback_artifacts import capture_rollout, load_policy
from generate_layout_preview import three_view_frames


OUTCOME_SPECS: Final = (
    (
        "success",
        (
            100001,
            100005,
            100006,
            100009,
            100012,
            100013,
            100016,
            100017,
            100020,
            100021,
            100022,
            100023,
            100027,
            100028,
            100029,
            100033,
            100039,
            100040,
            100041,
            100043,
            100046,
            100047,
            100048,
            100050,
            100051,
            100053,
            100061,
        ),
        True,
    ),
    (
        "failure",
        (
            100002,
            100007,
            100008,
            100010,
            100019,
            100034,
            100035,
            100045,
            100052,
            100055,
            100056,
            100060,
            100070,
            100073,
            100078,
        ),
        False,
    ),
)


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
    for outcome, seeds, expected_success in OUTCOME_SPECS:
        output = assets / f"2026-08-20_dp_transformer_{outcome}_three_view_120s_4x.mp4"
        with iio.get_writer(output, fps=40, macro_block_size=1) as writer:
            for number, seed in enumerate(seeds, start=1):
                capture = capture_rollout(
                    policy,
                    identity,
                    seed,
                    f"{outcome.upper()} {number:02d} / {len(seeds):02d}",
                )
                if capture.success != expected_success:
                    raise RuntimeError(f"seed {seed} no longer matches {outcome}")
                for frame in three_view_frames(capture):
                    writer.append_data(frame)
                del capture
                gc.collect()
        print(f"published {outcome} 3-view video under {assets}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
