"""Bind audited physical receipt digests into a fresh hardware profile."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from .policy_approval import ProductionTrustStore, RsaPkcs1v15Sha256Anchor
from .policy_parser import load_production_safety_policy
from .replay_receipts import require_digest, validate_camera_receipt, validate_joint_receipt


def _mapping(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a JSON mapping")
    return cast("dict[str, object]", raw)


@dataclass(frozen=True, slots=True)
class HardwareProfileBindingRequest:
    template: Path
    lineage: Path
    joint_receipt: Path
    camera_receipt: Path
    policy: Path
    trust_anchor: Path
    output: Path
    action_bridge_audited: bool


def bind_hardware_profile(request: HardwareProfileBindingRequest) -> dict[str, object]:
    """Validate physical receipts and produce one fresh content-bound profile."""
    template = request.template
    lineage = request.lineage
    joint_receipt = request.joint_receipt
    camera_receipt = request.camera_receipt
    policy = request.policy
    trust_anchor = request.trust_anchor
    output = request.output
    action_bridge_audited = request.action_bridge_audited
    if output.exists():
        raise ValueError("bound hardware profile output must be fresh")
    lineage_document = _mapping(lineage)
    lineage_digest = require_digest(lineage_document.get("lineage_digest"), "lineage digest")
    if lineage_document.get("valid") is not True:
        raise ValueError("compact lineage is not valid")
    joint_document = _mapping(joint_receipt)
    joint_digest = validate_joint_receipt(joint_document, expected_digest=None)
    camera_document = _mapping(camera_receipt)
    camera_digest = validate_camera_receipt(
        camera_document,
        expected_digest=None,
        expected_scope="authorized_physical_diagnostic",
    )
    trust = ProductionTrustStore.from_owner_anchors(
        (RsaPkcs1v15Sha256Anchor.from_pem_file(trust_anchor),)
    )
    approved_policy = load_production_safety_policy(policy, trust_store=trust)
    if not action_bridge_audited:
        raise ValueError("--action-bridge-audited is required after the bridge test gate")
    raw: object = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("hardware profile template is invalid")
    profile = dict(cast("Mapping[str, object]", raw))
    sim_raw = profile.get("sim_to_real")
    if not isinstance(sim_raw, Mapping):
        raise TypeError("hardware profile sim_to_real section is invalid")
    sim_to_real = dict(cast("Mapping[str, object]", sim_raw))
    profile["sim_to_real"] = sim_to_real
    sim_to_real.update(
        {
            "physical_camera_registration_calibrated": True,
            "action_bridge": "audited",
            "lineage_digest": lineage_digest,
            "policy_digest": approved_policy.canonical_digest,
            "camera_registration_digest": camera_digest,
            "joint_equivalence_digest": joint_digest,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return {
        "output": str(output.resolve()),
        "lineage_digest": lineage_digest,
        "policy_digest": approved_policy.canonical_digest,
        "camera_registration_digest": camera_digest,
        "joint_equivalence_digest": joint_digest,
        "action_bridge": "audited",
    }
