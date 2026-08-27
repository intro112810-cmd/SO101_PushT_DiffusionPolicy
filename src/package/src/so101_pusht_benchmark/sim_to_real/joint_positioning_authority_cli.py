"""Prepare and verify owner-signed manual-positioning read authority bytes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from .joint_positioning_authority import (
    authority_document,
    load_joint_positioning_authority,
)
from .policy_approval import ProductionTrustStore, RsaPkcs1v15Sha256Anchor
from .read_only_authority import canonical_authority_bytes, load_read_only_acquisition_authority
from .receipt_routing import (
    locate_receipt_path,
    prepare_receipt_directory,
    validate_receipt_identity,
)
from .secure_io import atomic_write_new, read_regular_leaf


def _trust(path: Path) -> ProductionTrustStore:
    anchor = RsaPkcs1v15Sha256Anchor.from_pem_file(path)
    return ProductionTrustStore.from_owner_anchors((anchor,))


def _publish(path: Path, content: bytes) -> None:
    location = validate_receipt_identity(locate_receipt_path(path), production=True)
    atomic_write_new(
        location.resolved.parent,
        location.resolved.name,
        content,
        temporary=f".{path.name}.positioning-{os.getpid()}.tmp",
    )


def _read_production(path: Path) -> bytes:
    location = validate_receipt_identity(locate_receipt_path(path), production=True)
    content, _ = read_regular_leaf(location.resolved.parent, location.resolved.name)
    return content


def prepare(
    *,
    acquisition_authority: Path,
    acquisition_signature: Path,
    trust_anchor: Path,
    authority_id: str,
    output_dir: Path,
) -> dict[str, object]:
    """Emit exact public bytes for out-of-process owner signing."""
    trust = _trust(trust_anchor)
    base = load_read_only_acquisition_authority(
        acquisition_authority,
        signature_path=acquisition_signature,
        trust_store=trust,
    )
    anchor = RsaPkcs1v15Sha256Anchor.from_pem_file(trust_anchor)
    document = authority_document(
        base,
        authority_id=authority_id,
        approved_by=anchor.signer_id,
        valid_from=datetime.now(timezone.utc).replace(microsecond=0),
    )
    encoded = canonical_authority_bytes(document)
    prepare_receipt_directory(output_dir, production=True)
    candidate = output_dir / "manual-positioning-authority.candidate.json"
    _publish(candidate, encoded)
    return {
        "candidate_path": str(candidate),
        "candidate_sha256": hashlib.sha256(encoded).hexdigest(),
        "sign_command": (
            'openssl dgst -sha256 -sign "$OWNER_PRIVATE_KEY" '
            f'-out "{output_dir / "manual-positioning-authority.sig"}" "{candidate}"'
        ),
        "hardware_accessed": False,
        "genuine_scope_granted": False,
    }


@dataclass(frozen=True, slots=True)
class AssembleRequest:
    candidate: Path
    signature: Path
    acquisition_authority: Path
    acquisition_signature: Path
    trust_anchor: Path
    output: Path


def assemble(request: AssembleRequest) -> dict[str, object]:
    """Verify all bindings and signature before immutable final publication."""
    trust = _trust(request.trust_anchor)
    base = load_read_only_acquisition_authority(
        request.acquisition_authority,
        signature_path=request.acquisition_signature,
        trust_store=trust,
    )
    verified = load_joint_positioning_authority(
        request.candidate,
        signature_path=request.signature,
        trust_store=trust,
        base=base,
    )
    encoded = _read_production(request.candidate)
    _publish(request.output, encoded)
    return {
        "authority_path": str(request.output),
        "signature_path": str(request.signature),
        "authority_digest": verified.canonical_digest,
        "expires_at": verified.expires_at.isoformat(),
        "owner_signature_verified": True,
        "hardware_accessed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--acquisition-authority", required=True, type=Path)
    prepare_parser.add_argument("--acquisition-signature", required=True, type=Path)
    prepare_parser.add_argument("--trust-anchor", required=True, type=Path)
    prepare_parser.add_argument("--authority-id", required=True)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--candidate", required=True, type=Path)
    assemble_parser.add_argument("--signature", required=True, type=Path)
    assemble_parser.add_argument("--acquisition-authority", required=True, type=Path)
    assemble_parser.add_argument("--acquisition-signature", required=True, type=Path)
    assemble_parser.add_argument("--trust-anchor", required=True, type=Path)
    assemble_parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare(
                acquisition_authority=args.acquisition_authority,
                acquisition_signature=args.acquisition_signature,
                trust_anchor=args.trust_anchor,
                authority_id=args.authority_id,
                output_dir=args.output_dir,
            )
        else:
            result = assemble(
                AssembleRequest(
                    args.candidate,
                    args.signature,
                    args.acquisition_authority,
                    args.acquisition_signature,
                    args.trust_anchor,
                    args.output,
                )
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
