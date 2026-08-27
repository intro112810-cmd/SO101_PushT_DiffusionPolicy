"""External deterministic authority for the complete frozen 400k DP-CNN route."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from .lineage_documents import DocumentInputs, validate_documents
from .lineage_io import hash_file, resolve_members, safe_absolute_file, scoped_roots
from .lineage_publish import prepare_output, publish_output
from .lineage_manifest import parse_manifest, validate_identity
from .lineage_runtime import BOUNDARY, observe_runtime, validate_runtime_fingerprint
from .lineage_sources import validate_installed_origins
from .lineage_types import (
    DEFAULT_LINEAGE_MANIFEST,
    DEFAULT_ROOTS,
    RECEIPT_SCHEMA,
    ArtifactAuthorityReceipt,
    AuthorityMemberReceipt,
    LineageError,
    LineageMember,
    LineageRoots,
    canonical_json,
)

__all__ = (
    "DEFAULT_LINEAGE_MANIFEST",
    "ArtifactAuthorityReceipt",
    "LineageError",
    "LineageRoots",
    "validate_lineage",
    "validate_lineage_to_file",
)


@dataclass(frozen=True, slots=True)
class _ResolvedInputs:
    manifest: Path
    document: dict[str, object]
    identity: dict[str, object]
    members: tuple[LineageMember, ...]
    paths: dict[str, Path]
    roots: LineageRoots
    runtime_fingerprint: dict[str, object]


def _validate_parsed(artifact_id: str, inputs: _ResolvedInputs) -> ArtifactAuthorityReceipt:
    if (
        inputs.document["artifact_id"] != artifact_id
        or inputs.identity["artifact_id"] != artifact_id
    ):
        raise LineageError("requested artifact identity mismatch")
    validate_identity(inputs.identity, artifact_id)
    validate_installed_origins(inputs.roots)
    receipts: list[AuthorityMemberReceipt] = []
    digests: dict[str, str] = {}
    for member in inputs.members:
        actual, size = hash_file(inputs.paths[member.label], member.label)
        if actual != member.sha256:
            raise LineageError(f"lineage member digest mismatch: {member.label}")
        digests[member.label] = actual
        receipts.append(
            AuthorityMemberReceipt(member.label, member.scope, member.path, actual, size)
        )
    if (
        inputs.identity["bundle_sha256"] != digests["policy"]
        or inputs.identity["source_checkpoint_sha256"] != digests["source_checkpoint"]
    ):
        raise LineageError("bundle/source immutable identity mismatch")
    if (
        inputs.identity["runtime_lock_digest"] != digests["runtime_lock"]
        or inputs.identity["frozen_environment_manifest_digest"]
        != digests["frozen_environment_manifest"]
    ):
        raise LineageError("runtime/upstream immutable identity mismatch")
    observation = observe_runtime(inputs.roots)
    validate_runtime_fingerprint(inputs.runtime_fingerprint, observation.fingerprint)
    validate_documents(
        DocumentInputs(
            inputs.paths,
            digests,
            inputs.identity,
            inputs.members,
            inputs.roots,
            artifact_id,
            observation.sources,
        )
    )
    ordered = tuple(sorted(receipts, key=lambda item: item.label))
    manifest_digest, _ = hash_file(inputs.manifest, "lineage authority manifest")
    dependency_boundary = BOUNDARY
    payload = {
        "artifact_id": artifact_id,
        "authority_manifest_sha256": manifest_digest,
        "dependency_boundary": dependency_boundary,
        "runtime_fingerprint": observation.fingerprint,
        "identity": inputs.identity,
        "members": [asdict(item) for item in ordered],
    }
    authority_digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return ArtifactAuthorityReceipt(
        RECEIPT_SCHEMA,
        True,
        artifact_id,
        manifest_digest,
        authority_digest,
        dependency_boundary,
        observation.fingerprint,
        inputs.identity.copy(),
        ordered,
    )


def _inputs(artifact_root: Path, manifest_path: Path, roots: LineageRoots) -> _ResolvedInputs:
    manifest = safe_absolute_file(manifest_path, "lineage authority manifest")
    document, identity, members, runtime_fingerprint = parse_manifest(manifest)
    authority_roots = scoped_roots(artifact_root, roots)
    paths = resolve_members(members, authority_roots, manifest)
    return _ResolvedInputs(
        manifest,
        document,
        identity,
        members,
        paths,
        roots,
        runtime_fingerprint,
    )


def validate_lineage(
    artifact_root: Path,
    artifact_id: str,
    *,
    manifest_path: Path = DEFAULT_LINEAGE_MANIFEST,
    roots: LineageRoots = DEFAULT_ROOTS,
) -> ArtifactAuthorityReceipt:
    """Validate every immutable selector/source/member and semantic identity."""
    return _validate_parsed(artifact_id, _inputs(artifact_root, manifest_path, roots))


def validate_lineage_to_file(
    artifact_root: Path,
    artifact_id: str,
    output: Path,
    *,
    manifest_path: Path = DEFAULT_LINEAGE_MANIFEST,
    roots: LineageRoots = DEFAULT_ROOTS,
) -> ArtifactAuthorityReceipt:
    """Validate and publish atomically through a pinned no-follow directory fd."""
    inputs = _inputs(artifact_root, manifest_path, roots)
    target = prepare_output(output, (*inputs.paths.values(), inputs.manifest))
    try:
        receipt = _validate_parsed(artifact_id, inputs)
        publish_output(target, receipt.to_bytes())
        return receipt
    finally:
        target.close()
