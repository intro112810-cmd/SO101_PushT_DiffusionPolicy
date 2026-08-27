#!/usr/bin/env python3
"""Build a guarded execution package from one verified shadow ledger cycle."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from so101_pusht_benchmark.sim_to_real.execution_package_builder import build_execution_package
from so101_pusht_benchmark.sim_to_real.ledger_chain import verify_ledger
from so101_pusht_benchmark.sim_to_real.ledger_io import load_ledger_documents
from so101_pusht_benchmark.sim_to_real.receipt_routing import prepare_receipt_directory
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutViolation


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", required=True, type=Path)
    p.add_argument("--cycle", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--previous-evidence-digest")
    a = p.parse_args()
    try:
        records = load_ledger_documents(a.ledger)
        verify_ledger(records)
        package = build_execution_package(
            records, cycle=a.cycle, previous_evidence_digest=a.previous_evidence_digest
        )
        prepare_receipt_directory(a.output.parent, production=True)
        a.output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, RolloutViolation) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(package, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
