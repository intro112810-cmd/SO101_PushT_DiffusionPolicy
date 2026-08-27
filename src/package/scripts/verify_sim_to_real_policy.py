#!/usr/bin/env python3
"""Verify one policy without publishing authority or touching hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutViolation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    try:
        policy = load_fixture_safety_policy(args.policy)
    except RolloutViolation as exc:
        print(str(exc), file=sys.stderr)
        return 2
    summary = {
        "accepted": True,
        "canonical_digest": policy.canonical_digest,
        "policy_id": policy.policy_id,
        "type": type(policy).__name__,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
