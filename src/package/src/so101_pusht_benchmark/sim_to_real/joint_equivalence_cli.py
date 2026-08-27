"""Explicit synthetic and governed-physical joint-equivalence CLI routing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import cast

from .joint_equivalence import audit_corpus_file, audit_production_corpus_file
from .policy_approval import ProductionTrustStore, RsaPkcs1v15Sha256Anchor
from .receipt_routing import (
    locate_receipt_path,
    prepare_receipt_directory,
    validate_receipt_identity,
    validate_receipt_path,
)
from .secure_io import atomic_write_new, read_regular_leaf, unlink_owned_leaf


def canonical_receipt_bytes(receipt: dict[str, object]) -> bytes:
    """Return the one accepted lexical JSON representation."""
    return (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-fixture", action="store_true")
    mode.add_argument("--governed-physical", action="store_true")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--corpus-authority", type=Path)
    parser.add_argument("--trust-anchor", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def publish_joint_receipt(path: Path, receipt: dict[str, object], *, production: bool) -> None:
    """Publish canonical bytes atomically, then verify path, inode, and bytes."""
    location = validate_receipt_identity(locate_receipt_path(path), production=production)
    prepare_receipt_directory(location.lexical.parent, production=production)
    location = validate_receipt_identity(locate_receipt_path(path), production=production)
    encoded = canonical_receipt_bytes(receipt)
    identity = atomic_write_new(
        location.resolved.parent,
        location.resolved.name,
        encoded,
        temporary=f".{location.resolved.name}.joint-{os.getpid()}.tmp",
    )
    accepted = False
    try:
        validate_receipt_identity(locate_receipt_path(path), production=production)
        persisted, info = read_regular_leaf(location.resolved.parent, location.resolved.name)
        if (info.st_dev, info.st_ino) != identity.inode or persisted != encoded:
            raise JointReceiptPublicationError("joint receipt verified IO drift")
        parsed: object = json.loads(persisted)
        if (
            not isinstance(parsed, dict)
            or canonical_receipt_bytes(cast("dict[str, object]", parsed)) != persisted
        ):
            raise JointReceiptPublicationError("joint receipt is not canonical JSON")
        accepted = True
    finally:
        if not accepted:
            unlink_owned_leaf(identity)


class JointReceiptPublicationError(ValueError):
    """Canonical publication failed after audit but before acceptance."""


def _production_authority(args: argparse.Namespace) -> Path:
    authority = args.corpus_authority
    if not isinstance(authority, Path):
        raise JointReceiptPublicationError("--governed-physical requires --corpus-authority")
    return authority


def run_joint_equivalence_cli(
    argv: list[str] | None = None,
    *,
    trust_store: ProductionTrustStore | None = None,
) -> int:
    """Audit one explicit mode and publish only after all authority checks pass."""
    args = _parser().parse_args(argv)
    production = bool(args.governed_physical)
    try:
        validate_receipt_path(args.output, production=production)
        if production:
            authority = _production_authority(args)
            if trust_store is None and isinstance(args.trust_anchor, Path):
                anchor = RsaPkcs1v15Sha256Anchor.from_pem_file(args.trust_anchor)
                trust_store = ProductionTrustStore.from_owner_anchors((anchor,))
            if trust_store is None:
                raise JointReceiptPublicationError(
                    "owner-governed production trust store is unavailable"
                )
            receipt = audit_production_corpus_file(
                args.corpus,
                args.policy,
                authority,
                trust_store=trust_store,
            )
        else:
            if args.corpus_authority is not None or args.trust_anchor is not None:
                raise JointReceiptPublicationError(
                    "synthetic mode cannot accept corpus authority or trust anchor"
                )
            receipt = audit_corpus_file(args.corpus, args.policy)
        publish_joint_receipt(args.output, receipt, production=production)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(canonical_receipt_bytes(receipt).decode(), end="")
    return 0


def main() -> int:
    """Default process has no production anchors and fails closed in physical mode."""
    return run_joint_equivalence_cli()
