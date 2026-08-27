"""Strict loader for one planner-produced production execution package."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import cast
from .arming import ArmingResult
from .authorization import AuthorizationToken
from .physical_ik_replay import parse_physical_ik_proposal
from .production_single_step import PreparedProductionSingleStep
from .rollout_codes import RolloutCode, RolloutViolation
from .single_step_authorization import SingleStepAuthorization


@dataclass(frozen=True, slots=True)
class ProductionPackageDocument:
    prepared: PreparedProductionSingleStep


def load_production_package(
    path: Path, authorization: SingleStepAuthorization, armed: ArmingResult
) -> ProductionPackageDocument:
    """Parse proposal, token, and genuine pre-history from exact JSON bytes."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, "production package missing") from exc
    if not isinstance(raw, dict):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "production package fields")
    document = cast("dict[str, object]", raw)
    if (
        set(document) != {"schema", "proposal", "token", "pre_sample_digests", "newer_than"}
        or document.get("schema") != "production-single-step-package-v1"
    ):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "production package fields")
    proposal_raw = document["proposal"]
    token_raw = document["token"]
    pre_raw = document["pre_sample_digests"]
    newer = document["newer_than"]
    if (
        not isinstance(proposal_raw, Mapping)
        or not isinstance(token_raw, Mapping)
        or not isinstance(pre_raw, list)
        or isinstance(newer, bool)
        or not isinstance(newer, (int, float))
        or not math.isfinite(float(newer))
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "production package content")
    proposal_map = cast("Mapping[str,object]", proposal_raw)
    declared = proposal_map.get("proposal_hash")
    if not isinstance(declared, str):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "proposal hash")
    proposal = parse_physical_ik_proposal(
        {k: v for k, v in proposal_map.items() if k != "proposal_hash"}, declared_hash=declared
    )
    token_map = cast("Mapping[str,object]", token_raw)
    try:
        token = AuthorizationToken(
            str(token_map["token_id"]),
            str(token_map["proposal_hash"]),
            str(token_map["policy_digest"]),
            str(token_map["command_id"]),
            float(cast("int|float", token_map["valid_until"])),
            str(token_map["digest"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "authorization token") from exc
    pre = frozenset(str(value) for value in cast("list[object]", pre_raw))
    if len(pre) < 2 or any(len(value) != 64 for value in pre):
        raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "pre-sample digests")
    return ProductionPackageDocument(
        PreparedProductionSingleStep(authorization, armed, token, proposal, pre, float(newer))
    )
