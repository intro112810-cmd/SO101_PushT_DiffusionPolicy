"""Guard one shadow campaign as ready for a motor-write-forbidden single step."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from so101_pusht_benchmark.sim_to_real.arming import ArmingCheckInput, check_arming
from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    ReceiptRoutingError,
    prepare_receipt_directory,
    validate_receipt_path,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutViolation


def _parse_now(value: str | None) -> datetime:
    if value is None:
        return datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit("--now must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SystemExit("--now must include a timezone")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--shadow-ledger", required=True, type=Path)
    parser.add_argument("--authorization", required=False, type=Path)
    parser.add_argument("--operational-evidence", required=False, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="durable arming receipt")
    parser.add_argument("--now", required=False, type=str)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = check_arming(
            ArmingCheckInput(
                profile_path=args.profile,
                policy_path=args.policy,
                shadow_ledger_path=args.shadow_ledger,
                authorization_path=args.authorization,
                operational_evidence_path=args.operational_evidence,
                now=_parse_now(args.now),
            )
        )
        output = validate_receipt_path(args.output, production=False)
        prepare_receipt_directory(output.parent, production=False)
    except (ReceiptRoutingError, RolloutViolation) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    document = {
        "armed": result.armed,
        "authorization_digest": result.authorization_digest,
        "command_id": result.command_id,
        "evidence_scope": "test_fixture_only",
        "motor_writes_performed": result.motor_writes_performed,
        "policy_digest": result.policy_digest,
        "policy_evidence": "fixture_authorization_not_production_authority",
        "proposal_hash": result.proposal_hash,
        "receipt_digest": result.receipt_digest,
    }
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
