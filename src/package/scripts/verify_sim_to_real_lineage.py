#!/usr/bin/env python3
"""Verify and publish deterministic authority for the frozen 400k DP-CNN lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from so101_pusht_benchmark.sim_to_real.lineage import (
    DEFAULT_LINEAGE_MANIFEST,
    LineageError,
    validate_lineage_to_file,
)
from so101_pusht_benchmark.sim_to_real.receipt_routing import (
    ReceiptRoutingError,
    locate_receipt_path,
    validate_receipt_identity,
)
from so101_pusht_benchmark.sim_to_real.replay_types import ARTIFACT_ROOT


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", default=DEFAULT_LINEAGE_MANIFEST, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        production = args.artifact_root.resolve(strict=False) == ARTIFACT_ROOT.resolve(strict=False)
        output = validate_receipt_identity(locate_receipt_path(args.output), production=production)
        receipt = validate_lineage_to_file(
            args.artifact_root,
            args.artifact,
            output.resolved,
            manifest_path=args.manifest,
        )
        validate_receipt_identity(output, production=production)
    except (LineageError, OSError, ReceiptRoutingError) as error:
        print(json.dumps({"error": str(error), "valid": False}, sort_keys=True), file=sys.stderr)
        return 2
    print(receipt.to_bytes().decode(), end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
