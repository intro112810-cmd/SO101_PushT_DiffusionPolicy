#!/usr/bin/env python3
"""Bind audited production receipts into a fresh SO-101 hardware profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from so101_pusht_benchmark.sim_to_real.hardware_profile_binding import (
    HardwareProfileBindingRequest,
    bind_hardware_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--joint-receipt", required=True, type=Path)
    parser.add_argument("--camera-receipt", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--trust-anchor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-bridge-audited", action="store_true")
    args = parser.parse_args()
    try:
        result = bind_hardware_profile(
            HardwareProfileBindingRequest(
                args.template,
                args.lineage,
                args.joint_receipt,
                args.camera_receipt,
                args.policy,
                args.trust_anchor,
                args.output,
                args.action_bridge_audited,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
