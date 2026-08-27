"""Strict bounded fixture and policy-budget boundary parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

from .bounded_authorization import BoundedAuthorization
from .policy_types import FixtureApprovedSafetyPolicy, ProductionApprovedSafetyPolicy
from .replay_receipts import parse_sample_document
from .rollout_codes import RolloutCode, RolloutViolation

SamplePair = tuple[dict[str, object], dict[str, object]]


@dataclass(frozen=True, slots=True)
class BoundedCycle:
    samples: SamplePair
    scene_pose: dict[str, object]


def _mapping(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, "bounded manifest missing") from exc
    if not isinstance(raw, dict):
        raise RolloutViolation(RolloutCode.R_MISSING, "bounded manifest mapping")
    return cast("dict[str, object]", raw)


def load_bounded_cycles(path: Path) -> tuple[str, tuple[BoundedCycle, ...]]:
    """Parse every cycle as exactly two content-addressed samples."""
    manifest = _mapping(path / "manifest.json")
    mode, raw_cycles = manifest.get("mode"), manifest.get("cycles")
    if not isinstance(mode, str) or not isinstance(raw_cycles, list):
        raise RolloutViolation(RolloutCode.R_MISSING, "bounded manifest fields")
    result: list[BoundedCycle] = []
    for raw_cycle in cast("list[object]", raw_cycles):
        if not isinstance(raw_cycle, dict):
            raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "cycle samples")
        cycle = cast("dict[str, object]", raw_cycle)
        if not isinstance(cycle.get("samples"), list):
            raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "cycle samples")
        samples = cast("list[object]", cycle["samples"])
        if len(samples) != 2 or any(not isinstance(item, dict) for item in samples):
            raise RolloutViolation(RolloutCode.HISTORY_INCOMPLETE, "cycle needs two samples")
        parsed = tuple(parse_sample_document(cast("dict[str, object]", item)) for item in samples)
        raw_pose = cycle.get("scene_pose")
        if not isinstance(raw_pose, dict):
            raise RolloutViolation(RolloutCode.R_MISSING, "cycle scene pose")
        result.append(
            BoundedCycle(
                cast("SamplePair", parsed),
                cast("dict[str, object]", raw_pose),
            )
        )
    return mode, tuple(result)


def check_bounded_budgets(
    authorization: BoundedAuthorization,
    policy: FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy,
) -> None:
    """Prevent signed authority from widening any approved policy budget."""
    budget = policy.bounded_rollout
    if (
        authorization.max_commands > budget.max_commands
        or authorization.max_duration_seconds > budget.max_duration_seconds
        or authorization.max_path_length_m > budget.max_path_length_m
        or authorization.max_error_count > budget.max_error_count
    ):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "authorization widens policy")
