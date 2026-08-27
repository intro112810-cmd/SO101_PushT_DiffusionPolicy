"""Content-addressed raw camera/table/checkpoint registration auditor."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from .camera_authority import verify_production_camera_authority
from .camera_corpus import parse_camera_corpus
from .camera_geometry import evaluate_correspondences, invalid, mapping
from .policy_approval import ProductionTrustStore
from .policy_parser import load_fixture_safety_policy, load_production_safety_policy
from .policy_types import (
    CameraPolicy,
    FixtureApprovedSafetyPolicy,
    ProductionApprovedSafetyPolicy,
)
from .read_only_authority import ReadOnlyCameraPolicy
from .rollout_codes import RolloutCode, RolloutViolation


def audit_camera_registration(
    corpus: Mapping[str, object],
    *,
    corpus_root: Path,
    source_scope: str,
    thresholds: CameraPolicy | ReadOnlyCameraPolicy,
) -> dict[str, object]:
    """Recompute residuals without granting fixture or physical publication authority."""
    parsed = parse_camera_corpus(
        corpus,
        corpus_root,
        thresholds.min_correspondences,
        expected_scope=source_scope,
    )
    fit_rmse, fit_max, fit_sim_rmse, fit_sim_max, fit_depth = evaluate_correspondences(
        parsed.model,
        parsed.fit,
        label="fit",
        resolution=parsed.resolution,
    )
    held_rmse, held_max, held_sim_rmse, held_sim_max, held_depth = evaluate_correspondences(
        parsed.model,
        parsed.held_out,
        label="held_out",
        resolution=parsed.resolution,
    )
    maximum_error = max(fit_max, held_max)
    maximum_sim_error = max(fit_sim_max, held_sim_max)
    maximum_depth = max(fit_depth, held_depth)
    focal = min(parsed.model.intrinsics[0], parsed.model.intrinsics[4])
    physical_limit_m = thresholds.max_correspondence_error_px * maximum_depth / focal
    if (
        held_rmse > thresholds.max_reprojection_error_px
        or maximum_error > thresholds.max_correspondence_error_px
        or maximum_sim_error > physical_limit_m
    ):
        raise invalid("recomputed projection or physical/simulation residual exceeds policy")
    return {
        "audited": True,
        "code": None,
        "digest": parsed.digest,
        "source_evidence_scope": source_scope,
        "metrics_source": "recomputed_from_raw_points_and_matrices",
        "fit_correspondences": len(parsed.fit),
        "held_out_correspondences": len(parsed.held_out),
        "fit_reprojection_error_px": fit_rmse,
        "held_out_reprojection_error_px": held_rmse,
        "max_correspondence_error_px": maximum_error,
        "fit_physical_to_sim_residual_m": fit_sim_rmse,
        "held_out_physical_to_sim_residual_m": held_sim_rmse,
        "max_physical_to_sim_residual_m": maximum_sim_error,
        "physical_to_sim_policy_limit_m": physical_limit_m,
        "member_digests": {key: value["sha256"] for key, value in sorted(parsed.members.items())},
        "checkpoint_view_members": sorted(
            key for key, value in parsed.members.items() if value["role"] == "checkpoint_held_out"
        ),
        "device_hash": corpus.get("device_hash"),
        "config_hash": corpus.get("config_hash"),
        "orientation_hash": corpus.get("orientation_hash"),
    }


def load_corpus(path: Path) -> tuple[Mapping[str, object], Path]:
    """Load one corpus and retain the root needed to verify raw members."""
    document = path / "corpus.json" if path.is_dir() else path
    if not document.is_file():
        raise RolloutViolation(
            RolloutCode.CAMERA_UNREGISTERED,
            "camera corpus is absent; synthetic fixture cannot substitute for physical evidence",
        )
    try:
        raw = json.loads(document.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise invalid(f"camera corpus cannot be read: {exc}") from exc
    return mapping(raw, "camera corpus"), document.parent


def _audit_with_policy(
    corpus: Mapping[str, object],
    corpus_root: Path,
    *,
    source_scope: str,
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
) -> dict[str, object]:
    return audit_camera_registration(
        corpus,
        corpus_root=corpus_root,
        source_scope=source_scope,
        thresholds=policy.camera,
    )


def audit_corpus_file(corpus_path: Path, policy_path: Path) -> dict[str, object]:
    """Audit one explicitly synthetic corpus against fixture policy budgets."""
    policy = load_fixture_safety_policy(policy_path)
    corpus, corpus_root = load_corpus(corpus_path)
    receipt = _audit_with_policy(
        corpus,
        corpus_root,
        source_scope="synthetic_test_fixture",
        policy=policy,
    )
    return {
        **receipt,
        "evidence_scope": "synthetic_test_fixture",
        "genuine_physical_corpus": False,
        "policy_authority": type(policy).__name__,
        "policy_evidence": "fixture_policy_not_production_authority",
    }


def audit_production_corpus_file(
    corpus_path: Path,
    policy_path: Path,
    *,
    identity_path: Path,
    authority_path: Path,
    trust_store: ProductionTrustStore,
) -> dict[str, object]:
    """Audit raw physical evidence only after governed policy and identity verification."""
    policy = load_production_safety_policy(policy_path, trust_store=trust_store)
    corpus, corpus_root = load_corpus(corpus_path)
    authority, identity = verify_production_camera_authority(
        corpus,
        authority_path=authority_path,
        identity_path=identity_path,
        policy=policy,
        trust_store=trust_store,
    )
    receipt = _audit_with_policy(corpus, corpus_root, source_scope="production", policy=policy)
    return {
        **receipt,
        "evidence_scope": "authorized_physical_diagnostic",
        "genuine_physical_corpus": True,
        "policy_authority": type(policy).__name__,
        "policy_evidence": "owner_governed_production_policy",
        "policy_digest": policy.canonical_digest,
        "identity_digest": identity.identity_digest,
        "provider_digest": authority.provider_digest,
        "camera_device_digest": authority.camera_device_digest,
        "calibration_digest": authority.calibration_digest,
        "corpus_authority_approval_id": authority.approval_id,
        "corpus_authority_signer": authority.approved_by,
    }
