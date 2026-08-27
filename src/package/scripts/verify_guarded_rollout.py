"""Verify a completed guarded-rollout session without opening any device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutViolation
from so101_pusht_benchmark.sim_to_real.session_verifier import verify_guarded_session


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path, help="session evidence directory")
    parser.add_argument(
        "--verify-cleanup",
        action="store_true",
        help="require the session cleanup receipt to report no remaining resources",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        receipt = verify_guarded_session(args.session, verify_cleanup=args.verify_cleanup)
    except RolloutViolation as exc:
        print(f"{exc.code.value}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt.to_document(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
