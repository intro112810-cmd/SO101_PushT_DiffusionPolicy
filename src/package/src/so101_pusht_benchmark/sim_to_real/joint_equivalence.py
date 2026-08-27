"""Content-addressed multi-pose joint-frame and FK equivalence auditor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .joint_equivalence_affine import derive_affine_mapping
from .joint_equivalence_authority import (
    ProductionCorpusAuthority,
    load_production_corpus_authority,
)
from .joint_equivalence_corpus import (
    JointEquivalencePolicy,
    MemberDocument,
    ParsedJointCorpus,
    load_joint_corpus_documents,
    load_joint_corpus_manifest,
    load_joint_member_documents,
    parse_joint_corpus,
    unproven,
)
from .joint_equivalence_fk import verify_joint_domains_and_fk
from .joint_mapping import JOINT_ORDER
from .policy_approval import ProductionTrustStore
from .policy_parser import load_fixture_safety_policy, load_production_safety_policy
from .policy_types import FixtureApprovedSafetyPolicy
from .rollout_codes import RolloutCode, RolloutViolation


def _receipt(
    parsed: ParsedJointCorpus,
    policy: JointEquivalencePolicy,
    authority: ProductionCorpusAuthority | None,
) -> dict[str, object]:
    affine = derive_affine_mapping(parsed.fit, parsed.held_out, policy, parsed.claimed_order)
    fk = verify_joint_domains_and_fk(parsed.members, policy)
    synthetic = authority is None
    receipt: dict[str, object] = {
        "audited": True,
        "digest": parsed.digest,
        "corpus_digest": parsed.digest,
        "policy_digest": policy.canonical_digest,
        "evidence_scope": (
            "synthetic_test_fixture" if synthetic else "authorized_physical_diagnostic"
        ),
        "genuine_physical_evidence": not synthetic,
        "deployment_valid": not synthetic,
        "blockers": ["synthetic fixture is not genuine physical evidence"] if synthetic else [],
        "joint_order": list(JOINT_ORDER),
        "computed_joint_order": list(affine.joint_order),
        "computed_scales_rad_per_degree": list(affine.scales_rad_per_degree),
        "computed_zero_radians": list(affine.zero_radians),
        "fit_count": len(parsed.fit),
        "held_out_count": len(parsed.held_out),
        "task_plane_pose_count": sum(member.category == "task_plane" for member in parsed.members),
        "max_fk_residual_m": fk.maximum_residual_m,
        "fk_oracle": fk.oracle,
        "member_hashes": list(parsed.member_hashes),
    }
    if authority is not None:
        receipt.update(
            {
                "corpus_identity_digest": authority.identity_digest,
                "provider_digest": authority.provider_digest,
                "device_digest": authority.device_digest,
                "calibration_digest": authority.calibration_digest,
                "capture_id": authority.capture_id,
                "approved_by": authority.approved_by,
                "approval_id": authority.approval_id,
            }
        )
    return receipt


def audit_joint_equivalence(
    corpus: Mapping[str, object],
    *,
    member_documents: Sequence[MemberDocument],
    policy: FixtureApprovedSafetyPolicy,
) -> dict[str, object]:
    """Audit an explicitly synthetic fixture without production promotion."""
    if type(policy) is not FixtureApprovedSafetyPolicy:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "fixture policy required")
    parsed = parse_joint_corpus(corpus, member_documents, policy)
    if parsed.origin != "synthetic_test_fixture" or "production_bindings" in corpus:
        raise unproven("synthetic mode cannot consume physical or production corpus claims")
    return _receipt(parsed, policy, None)


def audit_corpus_file(corpus_path: Path, policy_path: Path) -> dict[str, object]:
    """Load and audit an explicitly synthetic corpus against fixture thresholds."""
    corpus, member_documents = load_joint_corpus_documents(corpus_path)
    policy = load_fixture_safety_policy(policy_path)
    return audit_joint_equivalence(corpus, member_documents=member_documents, policy=policy)


def audit_production_corpus_file(
    corpus_path: Path,
    policy_path: Path,
    authority_path: Path,
    *,
    trust_store: ProductionTrustStore,
) -> dict[str, object]:
    """Audit governed physical evidence only after policy and corpus authentication."""
    if type(trust_store) is not ProductionTrustStore or not trust_store.is_governed():
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "governed trust store required")
    document_path, corpus = load_joint_corpus_manifest(corpus_path)
    policy = load_production_safety_policy(policy_path, trust_store=trust_store)
    authority = load_production_corpus_authority(
        authority_path,
        corpus=corpus,
        policy=policy,
        trust_store=trust_store,
    )
    member_documents = load_joint_member_documents(document_path, corpus)
    parsed = parse_joint_corpus(corpus, member_documents, policy)
    if parsed.origin != "physical_read_only_capture":
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "authorized corpus origin is not physical"
        )
    return _receipt(parsed, policy, authority)
