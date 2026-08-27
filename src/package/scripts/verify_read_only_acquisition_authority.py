#!/usr/bin/env python3
"""Independently verify one production read-only acquisition authority."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from so101_pusht_benchmark.sim_to_real.policy_approval import (
    ProductionTrustStore,
    RsaPkcs1v15Sha256Anchor,
)
from so101_pusht_benchmark.sim_to_real.read_only_authority import (
    load_read_only_acquisition_authority,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--signature", required=True, type=Path)
    parser.add_argument("--trust-anchor", required=True, type=Path)
    parser.add_argument("--at")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        anchor = RsaPkcs1v15Sha256Anchor.from_pem_file(args.trust_anchor)
        trust_store = ProductionTrustStore.from_owner_anchors((anchor,))
        now = datetime.fromisoformat(args.at.replace("Z", "+00:00")) if args.at else None
        authority = load_read_only_acquisition_authority(
            args.authority,
            signature_path=args.signature,
            trust_store=trust_store,
            now=now,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "valid": True,
                "schema": authority.schema,
                "artifact_scope": authority.artifact_scope,
                "authority_digest": authority.canonical_digest,
                "approved_by": authority.approved_by,
                "expires_at": authority.expires_at.isoformat(),
                "provider_digest": authority.provider_digest,
                "source_lineage_authority_digest": authority.source_lineage_authority_digest,
                "camera_permissions": authority.camera_permissions,
                "follower_permissions": authority.follower_permissions,
                "forbidden_capabilities": authority.forbidden_capabilities,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
