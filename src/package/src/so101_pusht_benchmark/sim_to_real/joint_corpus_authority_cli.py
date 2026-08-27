"""Prepare and assemble an owner-signed authority for one exact joint corpus."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
from dataclasses import dataclass

from .rsa_signing import public_key_from_private, rsa_pkcs1v15_sha256_sign

from .joint_equivalence_corpus import (
    digest,
    load_joint_corpus_documents,
    mapping,
    text,
)
from .policy_approval import ProductionTrustStore, RsaPkcs1v15Sha256Anchor
from .receipt_routing import (
    locate_receipt_path,
    prepare_receipt_directory,
    validate_receipt_identity,
)
from .secure_io import atomic_write_new, read_regular_leaf

_SCHEMA = "joint-equivalence-corpus-authority-v1"
_SCHEME = "rsa-pkcs1v15-sha256-v1"
_REQUEST_SCHEMA = "joint-equivalence-corpus-authority-request-v1"


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _publish(path: Path, content: bytes) -> None:
    location = validate_receipt_identity(locate_receipt_path(path), production=True)
    atomic_write_new(
        location.resolved.parent,
        location.resolved.name,
        content,
        temporary=f".{path.name}.authority-{os.getpid()}.tmp",
    )


def _anchor(path: Path) -> RsaPkcs1v15Sha256Anchor:
    return RsaPkcs1v15Sha256Anchor.from_pem_file(path)


def _read_production(path: Path) -> bytes:
    location = validate_receipt_identity(locate_receipt_path(path), production=True)
    content, _ = read_regular_leaf(location.resolved.parent, location.resolved.name)
    return content


def prepare_request(
    corpus_path: Path,
    trust_anchor: Path,
    approval_id: str,
    output_dir: Path,
) -> dict[str, object]:
    """Freeze exact member hashes and emit only public signing material."""
    corpus, members = load_joint_corpus_documents(corpus_path)
    if len(members) == 0 or corpus.get("evidence_origin") != "physical_read_only_capture":
        raise ValueError("only a completed physical read-only corpus can request authority")
    if corpus.get("publication_status") != "owner_signature_required":
        raise ValueError("corpus is not at the owner-signature publication boundary")
    bindings = mapping(corpus.get("production_bindings"), "production corpus bindings")
    anchor = _anchor(trust_anchor)
    identity: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact_scope": "authorized_physical_diagnostic",
        "approved_by": anchor.signer_id,
        "approval_id": approval_id,
        "corpus_digest": digest(corpus.get("corpus_digest"), "corpus digest"),
        "policy_digest": digest(corpus.get("policy_digest"), "policy digest"),
        "provider_digest": digest(bindings.get("provider_digest"), "provider digest"),
        "device_digest": digest(bindings.get("device_digest"), "device digest"),
        "calibration_digest": digest(bindings.get("calibration_digest"), "calibration digest"),
        "capture_id": text(bindings.get("capture_id"), "capture id"),
    }
    identity_digest = hashlib.sha256(_canonical(identity)).hexdigest()
    binding = {
        "approval_id": approval_id,
        "identity_digest": identity_digest,
        "schema": _SCHEMA,
        "signer_id": anchor.signer_id,
    }
    request = {
        "schema": _REQUEST_SCHEMA,
        "identity": identity,
        "identity_digest": identity_digest,
        "scheme": _SCHEME,
        "binding_sha256": hashlib.sha256(_canonical(binding)).hexdigest(),
        "exact_member_count": len(members),
        "exact_member_hashes": [entry[0]["sha256"] for entry in members],
    }
    prepare_receipt_directory(output_dir, production=True)
    request_path = output_dir / "corpus-authority-request.json"
    binding_path = output_dir / "corpus-authority-binding.json"
    _publish(request_path, _canonical(request) + b"\n")
    _publish(binding_path, _canonical(binding))
    return {
        "request_path": str(request_path),
        "binding_path": str(binding_path),
        "binding_sha256": request["binding_sha256"],
        "exact_member_count": len(members),
        "sign_command": (
            'openssl dgst -sha256 -sign "$OWNER_PRIVATE_KEY" '
            f'-out "{output_dir / "corpus-authority-binding.sig"}" "{binding_path}"'
        ),
        "genuine_scope_granted": False,
    }


def assemble_authority(
    request_path: Path,
    binding_path: Path,
    signature_path: Path,
    trust_anchor: Path,
    output: Path,
) -> dict[str, object]:
    """Verify the detached owner signature before canonical authority publication."""
    request_bytes = _read_production(request_path)
    binding_bytes = _read_production(binding_path)
    signature = _read_production(signature_path)
    try:
        request_raw: object = json.loads(request_bytes)
        binding_raw: object = json.loads(binding_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("authority request or binding is invalid") from exc
    request = mapping(request_raw, "authority request")
    binding = mapping(binding_raw, "authority binding")
    if request.get("schema") != _REQUEST_SCHEMA or _canonical(binding) != binding_bytes:
        raise ValueError("authority request/binding is noncanonical")
    if hashlib.sha256(binding_bytes).hexdigest() != request.get("binding_sha256"):
        raise ValueError("authority binding drift")
    anchor = _anchor(trust_anchor)
    identity = mapping(request.get("identity"), "authority request identity")
    identity_digest = digest(request.get("identity_digest"), "identity digest")
    if hashlib.sha256(_canonical(identity)).hexdigest() != identity_digest:
        raise ValueError("authority request identity drift")
    if not anchor.verify(
        text(identity.get("approved_by"), "approved_by"),
        _SCHEME,
        binding_bytes,
        signature.hex(),
    ):
        raise ValueError("owner corpus signature is untrusted")
    authority = {
        **dict(identity),
        "scheme": _SCHEME,
        "identity_digest": identity_digest,
        "binding_signature": signature.hex(),
    }
    _publish(output, _canonical(authority) + b"\n")
    trust = ProductionTrustStore.from_owner_anchors((anchor,))
    if not trust.is_governed():
        raise ValueError("governed trust store is unavailable")
    return {
        "authority_path": str(output),
        "authority_sha256": hashlib.sha256(_canonical(authority) + b"\n").hexdigest(),
        "corpus_digest": authority["corpus_digest"],
        "exact_member_count": request.get("exact_member_count"),
        "owner_signature_verified": True,
        "genuine_scope_granted": False,
        "next_step": "run the governed physical joint-equivalence auditor",
    }


@dataclass(frozen=True, slots=True)
class JointAuthorityIssueRequest:
    corpus_path: Path
    trust_anchor: Path
    private_key: Path
    approval_id: str
    output_dir: Path
    output: Path


def issue_authority(request: JointAuthorityIssueRequest) -> dict[str, object]:
    """Prepare, sign, verify, and publish one exact authority offline."""
    corpus_path = request.corpus_path
    trust_anchor = request.trust_anchor
    private_key = request.private_key
    approval_id = request.approval_id
    output_dir = request.output_dir
    output = request.output
    anchor = _anchor(trust_anchor)
    key = private_key.read_bytes()
    if hashlib.sha256(public_key_from_private(key)).hexdigest() != anchor.signer_id:
        raise ValueError("private key and trust anchor differ")
    prepared = prepare_request(corpus_path, trust_anchor, approval_id, output_dir)
    binding_path = Path(str(prepared["binding_path"]))
    signature_path = output_dir / "corpus-authority-binding.sig"
    _publish(signature_path, rsa_pkcs1v15_sha256_sign(key, _read_production(binding_path)))
    return assemble_authority(
        Path(str(prepared["request_path"])),
        binding_path,
        signature_path,
        trust_anchor,
        output,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="freeze exact public signing material")
    prepare.add_argument("--corpus", required=True, type=Path)
    prepare.add_argument("--trust-anchor", required=True, type=Path)
    prepare.add_argument("--approval-id", required=True)
    prepare.add_argument("--output-dir", required=True, type=Path)
    assemble = subparsers.add_parser("assemble", help="verify signature and publish authority")
    assemble.add_argument("--request", required=True, type=Path)
    assemble.add_argument("--binding", required=True, type=Path)
    assemble.add_argument("--signature", required=True, type=Path)
    assemble.add_argument("--trust-anchor", required=True, type=Path)
    assemble.add_argument("--output", required=True, type=Path)
    issue = subparsers.add_parser("issue", help="prepare, sign, and publish offline")
    issue.add_argument("--corpus", required=True, type=Path)
    issue.add_argument("--trust-anchor", required=True, type=Path)
    issue.add_argument("--private-key", required=True, type=Path)
    issue.add_argument("--approval-id", required=True)
    issue.add_argument("--output-dir", required=True, type=Path)
    issue.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare_request(
                args.corpus, args.trust_anchor, args.approval_id, args.output_dir
            )
        elif args.command == "assemble":
            result = assemble_authority(
                args.request,
                args.binding,
                args.signature,
                args.trust_anchor,
                args.output,
            )
        else:
            result = issue_authority(
                JointAuthorityIssueRequest(
                    args.corpus,
                    args.trust_anchor,
                    args.private_key,
                    args.approval_id,
                    args.output_dir,
                    args.output,
                )
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
