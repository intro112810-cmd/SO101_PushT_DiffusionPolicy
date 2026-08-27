#!/usr/bin/env python3
"""Issue signed camera live identity and exact corpus authority offline."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from so101_pusht_benchmark.sim_to_real.camera_authority_issuance import (
    CameraAuthorityIssuanceRequest,
    issue_camera_authorities,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base-authority",
        "base-signature",
        "policy",
        "trust-anchor",
        "private-key",
        "corpus",
        "output-dir",
    ):
        p.add_argument(f"--{name}", required=True, type=Path)
    p.add_argument("--approval-id", required=True)
    a = p.parse_args()
    try:
        result = issue_camera_authorities(
            CameraAuthorityIssuanceRequest(
                a.base_authority,
                a.base_signature,
                a.policy,
                a.trust_anchor,
                a.private_key,
                a.corpus,
                a.output_dir,
                a.approval_id,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
