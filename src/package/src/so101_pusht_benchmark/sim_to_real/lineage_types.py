"""Typed contracts for deterministic sim-to-real lineage authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sysconfig
from typing import Literal

Scope = Literal["artifact", "package", "project", "runtime"]
MANIFEST_SCHEMA = "so101-sim-to-real-lineage-manifest-v1"
RECEIPT_SCHEMA = "so101-artifact-authority-receipt-v1"
PACKAGE_ROOT = Path(__file__).parents[3]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_LINEAGE_MANIFEST = PACKAGE_ROOT / "configs/provenance/sim_to_real_400k_lineage.json"


class LineageError(RuntimeError):
    """Raised before authority publication when a lineage claim is untrusted."""


@dataclass(frozen=True, slots=True)
class LineageRoots:
    """Pinned roots used to resolve every manifest scope."""

    package: Path
    project: Path
    runtime: Path


DEFAULT_ROOTS = LineageRoots(
    PACKAGE_ROOT,
    PROJECT_ROOT,
    Path(sysconfig.get_paths()["purelib"]),
)


@dataclass(frozen=True, slots=True)
class LineageMember:
    """One parsed immutable member declaration."""

    label: str
    scope: Scope
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class AuthorityMemberReceipt:
    """One verified immutable member in an accepted authority receipt."""

    label: str
    scope: Scope
    path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class ArtifactAuthorityReceipt:
    """Typed accepted authority; invalid receipts cannot be constructed."""

    schema: str
    valid: bool
    artifact_id: str
    authority_manifest_sha256: str
    authority_digest: str
    dependency_boundary: str
    runtime_fingerprint: dict[str, object]
    identity: dict[str, object]
    members: tuple[AuthorityMemberReceipt, ...]

    def __post_init__(self) -> None:
        """Keep accepted receipt construction fail-closed."""
        if self.schema != RECEIPT_SCHEMA or self.valid is not True:
            raise ValueError("ArtifactAuthorityReceipt valid/schema invariant failed")
        expected_boundary = (
            "python_sources_and_loaded_extensions_exact_third_party_distributions_"
            "version_origin_bound_runtime_lock_hashed"
        )
        if self.dependency_boundary != expected_boundary:
            raise ValueError("ArtifactAuthorityReceipt dependency boundary is invalid")

    def to_bytes(self) -> bytes:
        """Return the canonical byte-stable receipt representation."""
        return pretty_json(asdict(self))


def pretty_json(value: object) -> bytes:
    """Encode deterministic human-readable JSON."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def canonical_json(value: object) -> bytes:
    """Encode deterministic compact JSON for authority digesting."""
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
