"""Semantic replay for durable planner-complete shadow decision ledgers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import cast

from .authorization import AuthorizationToken
from .ledger_chain import canonical_hash, verify_ledger
from .physical_ik import physical_ik_proposal_hash
from .physical_ik_replay import parse_physical_ik_proposal
from .replay_history import parse_inference_receipt, parse_sample_document
from .rollout_codes import RolloutCode, RolloutViolation
from .task_frame import TransformMaterial, apply_se2, parse_rigid_se2, se2_hash
from .task_frame_bridge import CartesianProposalReceipt

_FIXTURE_POLICY_EVIDENCE = "fixture_adapter_not_frozen_production"
_FROZEN_POLICY_EVIDENCE = "authentic_frozen_production"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"shadow ledger {label}")
    return cast("Mapping[str, object]", value)


def _items(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"shadow ledger {label}")
    return cast("list[object]", value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"shadow ledger {label}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"shadow ledger {label}")
    number = float(value)
    if not math.isfinite(number):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"shadow ledger {label}")
    return number


def _floats(value: object, count: int, label: str) -> tuple[float, ...]:
    items = _items(value, label)
    if len(items) != count:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"shadow ledger {label}")
    return tuple(_number(item, label) for item in items)


def _token(document: Mapping[str, object]) -> AuthorizationToken:
    return AuthorizationToken(
        token_id=_text(document.get("token_id"), "token_id"),
        proposal_hash=_text(document.get("proposal_hash"), "token proposal"),
        policy_digest=_text(document.get("policy_digest"), "token policy"),
        command_id=_text(document.get("command_id"), "token command"),
        valid_until=_number(document.get("valid_until"), "token expiry"),
        digest=_text(document.get("digest"), "token digest"),
    )


def _samples(
    record: Mapping[str, object],
) -> tuple[tuple[str, str], tuple[str, str], str]:
    raw_records = _items(record.get("sample_records"), "sample records")
    if len(raw_records) != 2:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "shadow ledger sample count")
    parsed = [parse_sample_document(_mapping(raw, "sample record")) for raw in raw_records]
    sample_ids = tuple(_text(sample["record_id"], "sample id") for sample in parsed)
    sample_digests = tuple(_text(sample["digest"], "sample digest") for sample in parsed)
    if list(sample_ids) != record.get("sample_ids"):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "sample id binding")
    if list(sample_digests) != record.get("sample_digests"):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "sample digest binding")
    policy_digest = _text(record.get("policy_digest"), "sample policy")
    return (
        cast("tuple[str, str]", sample_ids),
        cast("tuple[str, str]", sample_digests),
        policy_digest,
    )


def _cartesian(
    record: Mapping[str, object],
    inference_action: list[float],
    policy_digest: str,
) -> None:
    document = _mapping(record.get("cartesian_receipt"), "Cartesian receipt")
    if canonical_hash(document) != record.get("cartesian_receipt_hash"):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "Cartesian receipt hash")
    material_document = _mapping(record.get("transform_material"), "transform material")
    material = TransformMaterial(
        parse_rigid_se2(material_document.get("physical_to_sim_se2")),
        _text(material_document.get("camera_digest"), "transform camera digest"),
    )
    transform_hash = se2_hash(material)
    if (
        document.get("transform_hash") != transform_hash
        or record.get("transform_hash") != transform_hash
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "Cartesian transform hash")
    raw_xy = _floats(document.get("raw_xy"), 2, "raw_xy")
    raw_xyz = _floats(document.get("raw_xyz"), 3, "raw_xyz")
    applied_xyz = _floats(document.get("applied_xyz"), 3, "applied_xyz")
    tool_rpy = _floats(document.get("tool_rpy"), 3, "tool_rpy")
    if list(raw_xy) != inference_action:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "Cartesian action binding")
    expected_xy = apply_se2(material, cast("tuple[float, float]", raw_xy))
    if not all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(raw_xyz[:2], expected_xy, strict=True)
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "Cartesian transform replay")
    if raw_xyz != applied_xyz or document.get("clipping_performed") is not False:
        raise RolloutViolation(RolloutCode.R_CLIPPING_REQUIRED, "Cartesian applied_xyz")
    if document.get("ik_called") is not False or document.get("policy_digest") != policy_digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "Cartesian decision binding")
    CartesianProposalReceipt(
        cast("tuple[float, float]", raw_xy),
        cast("tuple[float, float, float]", raw_xyz),
        cast("tuple[float, float, float]", applied_xyz),
        cast("tuple[float, float, float]", tool_rpy),
        transform_hash,
        _text(document.get("camera_digest"), "Cartesian camera digest"),
        policy_digest,
        False,
        False,
    )


def _ik(record: Mapping[str, object], joint_digest: str) -> str:
    document = dict(_mapping(record.get("ik_proposal"), "IK proposal"))
    declared = _text(document.pop("proposal_hash", None), "IK proposal hash")
    if physical_ik_proposal_hash(document) != declared or record.get("proposal_hash") != declared:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "IK proposal hash")
    proposal = parse_physical_ik_proposal(document, declared_hash=declared)
    if proposal.joint_equivalence_digest != joint_digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "IK joint binding")
    return proposal.proposal_hash


def _cleanup(record: Mapping[str, object]) -> None:
    expected = {
        "status": "released",
        "writer_closed": True,
        "motor_writes_performed": False,
        "actuation_performed": False,
        "writer_symbols": 0,
        "read_only": True,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "cleanup semantics")


def verify_shadow_decision_ledger(records: Sequence[Mapping[str, object]]) -> str:
    """Recompute inner receipts and enforce complete cycle and cleanup semantics."""
    terminal_digest = verify_ledger(records)
    cycle: int | None = None
    stage = "none"
    sample_ids: tuple[str, str] | None = None
    sample_digests: tuple[str, str] | None = None
    policy_digest = ""
    joint_digest = ""
    action: list[float] | None = None
    proposal_hash: str | None = None
    decided: set[int] = set()
    sampled: set[int] = set()
    for record in records:
        kind = record.get("kind")
        record_cycle = record.get("cycle")
        if kind == "samples":
            if isinstance(record_cycle, bool) or not isinstance(record_cycle, int):
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "shadow ledger cycle")
            cycle = record_cycle
            sampled.add(cycle)
            sample_ids, sample_digests, policy_digest = _samples(record)
            stage = "samples"
        elif kind == "inference":
            if stage != "samples" or sample_ids is None or sample_digests is None:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "inference stage order")
            raw = dict(_mapping(record.get("inference_receipt"), "inference receipt"))
            raw["inference_digest"] = record.get("inference_digest")
            inference = parse_inference_receipt(raw)
            if inference.sample_ids != sample_ids or inference.sample_digests != sample_digests:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "inference sample binding")
            joint_digest = inference.joint_digest
            action = inference.action_chunk_float32_2d[0]
            if record.get("selected_action_0") != action:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "selected action zero")
            claim = (
                _FIXTURE_POLICY_EVIDENCE
                if inference.policy == "fixture_deterministic_adapter"
                else _FROZEN_POLICY_EVIDENCE
            )
            if record.get("policy_evidence") != claim:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "policy evidence claim")
            stage = "inference"
        elif kind == "cartesian_transform":
            if stage != "inference" or action is None:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "Cartesian stage order")
            _cartesian(record, action, policy_digest)
            stage = "cartesian_transform"
        elif kind == "ik_proposal":
            if stage != "cartesian_transform":
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "IK stage order")
            proposal_hash = _ik(record, joint_digest)
            stage = "ik_proposal"
        elif kind == "supervisor_decision":
            if cycle is None or record_cycle != cycle:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "supervisor cycle")
            if record.get("decision") == "ACCEPT":
                token = _token(_mapping(record.get("authorization_token"), "token"))
                if (
                    stage != "ik_proposal"
                    or token.proposal_hash != proposal_hash
                    or token.policy_digest != policy_digest
                ):
                    raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "supervisor binding")
            elif record.get("decision") == "REJECT":
                if record.get("after_stage") != stage:
                    raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "rejection stage")
                _text(record.get("rejection_code"), "rejection code")
                _text(record.get("rejection_detail"), "rejection detail")
            else:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "supervisor decision")
            decided.add(cycle)
        elif kind == "cleanup":
            _cleanup(record)
    if not sampled or sampled != decided:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "incomplete cycle decisions")
    return terminal_digest
