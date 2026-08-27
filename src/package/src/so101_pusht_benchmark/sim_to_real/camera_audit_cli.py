"""Explicit fixture or governed-physical camera audit CLI boundary."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import sys

from .camera_registration import audit_corpus_file, audit_production_corpus_file
from .policy_approval import ProductionTrustStore
from .receipt_routing import (
    locate_receipt_path,
    prepare_receipt_directory,
    ReceiptRoutingError,
    validate_receipt_identity,
)
from .rollout_codes import RolloutCode, RolloutViolation
from .secure_io import atomic_write_new, unlink_owned_leaf

ReceiptPublisher = Callable[[Path, dict[str, object], bool], None]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("fixture", "physical"))
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--corpus-authority", type=Path)
    parser.add_argument("--trust-anchor", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def publish_camera_receipt(path: Path, receipt: dict[str, object], production: bool) -> None:
    """Publish atomically only after lexical and resolved identities agree twice."""
    location = validate_receipt_identity(locate_receipt_path(path), production=production)
    prepare_receipt_directory(location.lexical.parent, production=production)
    location = validate_receipt_identity(locate_receipt_path(path), production=production)
    content = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    published = atomic_write_new(
        location.resolved.parent,
        location.resolved.name,
        content,
        temporary=f".{location.resolved.name}.camera-audit-{os.getpid()}.tmp",
    )
    try:
        validate_receipt_identity(locate_receipt_path(path), production=production)
    except (ReceiptRoutingError, RolloutViolation):
        unlink_owned_leaf(published)
        raise


def _physical_receipt(
    args: argparse.Namespace,
    trust_store: ProductionTrustStore | None,
) -> dict[str, object]:
    if trust_store is None:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED,
            "governed production trust store is unavailable",
        )
    if args.identity is None or args.corpus_authority is None:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED,
            "signed physical identity and corpus authority are required",
        )
    return audit_production_corpus_file(
        args.corpus,
        args.policy,
        identity_path=args.identity,
        authority_path=args.corpus_authority,
        trust_store=trust_store,
    )


def run_camera_audit_cli(
    argv: list[str] | None = None,
    *,
    production_trust_store: ProductionTrustStore | None = None,
    publisher: ReceiptPublisher = publish_camera_receipt,
) -> int:
    """Validate output authority before reading policy, corpus, or signed identities."""
    args = _parser().parse_args(argv)
    production = args.mode == "physical"
    try:
        output_identity = validate_receipt_identity(
            locate_receipt_path(args.output),
            production=production,
        )
        trust_store = production_trust_store
        if production and trust_store is None:
            if args.trust_anchor is None:
                raise RolloutViolation(
                    RolloutCode.R_POLICY_UNAUTHORIZED,
                    "physical audit trust store requires --trust-anchor",
                )
            from .policy_approval import RsaPkcs1v15Sha256Anchor

            trust_store = ProductionTrustStore.from_owner_anchors(
                (RsaPkcs1v15Sha256Anchor.from_pem_file(args.trust_anchor),)
            )
        receipt = (
            _physical_receipt(args, trust_store)
            if production
            else audit_corpus_file(args.corpus, args.policy)
        )
        validate_receipt_identity(output_identity, production=production)
        publisher(output_identity.lexical, receipt, production)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0
