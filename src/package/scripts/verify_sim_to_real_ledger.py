"""Verify one hash-chained rollout ledger and print its deterministic replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from so101_pusht_benchmark.sim_to_real.ledger_chain import (
    LedgerViolation,
    replay_digest,
    verify_ledger,
)
from so101_pusht_benchmark.sim_to_real.ledger_io import load_ledger_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--replay", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        documents = load_ledger_documents(args.ledger)
        valid_digest = verify_ledger(documents)
        if args.replay:
            result: dict[str, object] = {
                "valid": True,
                "record_count": len(documents),
                "terminal_digest": valid_digest,
                "replay_digest": replay_digest(documents),
            }
        else:
            result = {
                "valid": True,
                "record_count": len(documents),
                "terminal_digest": valid_digest,
            }
    except LedgerViolation as exc:
        print(f"ledger hash mismatch: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
