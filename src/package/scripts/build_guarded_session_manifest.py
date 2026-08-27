"""Finalize one exact content-addressed guarded-rollout session manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    locate_receipt_path,
    ReceiptRoutingError,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutViolation
from so101_pusht_benchmark.sim_to_real.session_manifest import write_session_manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path, help="complete session directory")
    parser.add_argument("--session-id", required=True, help="stable session identifier")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = write_session_manifest(
            args.session,
            session_id=args.session_id,
        )
    except (ReceiptRoutingError, RolloutViolation) as exc:
        print(f"R_MISSING: {exc}", file=sys.stderr)
        return 2
    manifest_io = locate_receipt_path(manifest).resolved
    document = json.loads(manifest_io.read_text(encoding="utf-8"))
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
