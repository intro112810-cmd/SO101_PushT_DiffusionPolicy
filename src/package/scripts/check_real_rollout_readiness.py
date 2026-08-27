"""Check the physical rollout gate without opening cameras or motor buses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from so101_pusht_benchmark.hardware_profile import (
    load_hardware_profile,
    rollout_readiness_blockers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--inference-receipt", required=True, type=Path)
    parser.add_argument("--confirm-low-speed-rollout", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_hardware_profile(args.profile.resolve())
    raw = json.loads(args.inference_receipt.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("inference receipt must be a mapping")
    receipt = cast("dict[str, object]", raw)
    blockers = rollout_readiness_blockers(
        profile,
        receipt,
        confirmed=args.confirm_low_speed_rollout,
    )
    report = {
        "rollout_ready": not blockers,
        "actuation_performed": False,
        "blockers": blockers,
    }
    print(json.dumps(report, indent=2))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
