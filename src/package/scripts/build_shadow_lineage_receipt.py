#!/usr/bin/env python3
"""Build a compact production lineage receipt for physical shadow inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from so101_pusht_benchmark.sim_to_real.shadow_lineage_builder import build_compact_lineage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = build_compact_lineage(args.source, args.output)
    except (OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
