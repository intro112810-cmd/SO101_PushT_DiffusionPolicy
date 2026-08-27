from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from sim_to_real_policy_helpers import (
    APPROVED,
    YamlValue,
    mapping,
    raw_policy,
    reject_policy,
)
from so101_pusht_benchmark.sim_to_real import (
    policy_canonical,
    policy_io,
    policy_schema,
    policy_types,
    policy_values,
)
from so101_pusht_benchmark.sim_to_real.policy_parser import (
    ProductionTrustStore,
    load_fixture_safety_policy,
    load_production_safety_policy,
    require_production_policy,
)
from so101_pusht_benchmark.sim_to_real.policy_types import (
    FixtureApprovedSafetyPolicy,
    ProductionApprovedSafetyPolicy,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation

ROOT = Path(__file__).parents[1]
PENDING = ROOT / "configs/hardware/sim_to_real_safety_policy_v1.pending.yaml"
NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
POLICY_SOURCE = ROOT / "src/so101_pusht_benchmark/sim_to_real"


def _integer_equivalent(value: YamlValue) -> YamlValue:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_integer_equivalent(item) for item in value]
    if isinstance(value, dict):
        return {key: _integer_equivalent(item) for key, item in value.items()}
    return value


def test_typed_canonical_digest_ignores_yaml_key_order(tmp_path: Path) -> None:
    original = load_fixture_safety_policy(APPROVED, now=NOW)
    raw = raw_policy()
    reordered = {key: raw[key] for key in reversed(raw)}
    path = tmp_path / "reordered.yaml"
    path.write_text(yaml.safe_dump(reordered, sort_keys=False), encoding="utf-8")

    parsed = load_fixture_safety_policy(path, now=NOW)
    assert parsed.canonical_content == original.canonical_content
    assert parsed.canonical_digest == original.canonical_digest


def test_every_integer_equivalent_float_spelling_has_one_typed_digest(tmp_path: Path) -> None:
    original = load_fixture_safety_policy(APPROVED, now=NOW)
    raw = raw_policy()
    normalized = _integer_equivalent(raw)
    assert isinstance(normalized, dict)
    path = tmp_path / "integer-spellings.yaml"
    path.write_text(yaml.safe_dump(normalized, sort_keys=False), encoding="utf-8")

    parsed = load_fixture_safety_policy(path, now=NOW)
    assert parsed.canonical_content == original.canonical_content
    assert parsed.canonical_digest == original.canonical_digest


def test_semantic_threshold_change_has_different_typed_content(tmp_path: Path) -> None:
    original = load_fixture_safety_policy(APPROVED, now=NOW)
    raw = raw_policy()
    thresholds = mapping(raw["thresholds"])
    watchdog = mapping(thresholds["watchdog"])
    watchdog["timeout_seconds"] = 0.3

    reject_policy(raw, tmp_path / "semantic-change.yaml", NOW)
    assert b'"timeout_seconds":0.3' not in original.canonical_content


def test_fixture_and_production_authority_are_nominally_separate() -> None:
    fixture = load_fixture_safety_policy(APPROVED, now=NOW)

    assert type(fixture) is FixtureApprovedSafetyPolicy
    assert FixtureApprovedSafetyPolicy is not ProductionApprovedSafetyPolicy
    for value in (fixture, raw_policy(), APPROVED, True, None):
        with pytest.raises(RolloutViolation) as caught:
            require_production_policy(value)
        assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED


def test_fixture_replay_and_pending_cannot_mint_production_policy(tmp_path: Path) -> None:
    trust_store = object.__new__(ProductionTrustStore)
    raw = raw_policy()
    raw["artifact_scope"] = "production"
    replay = tmp_path / "replay.yaml"
    replay.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    for path in (replay, PENDING):
        with pytest.raises(RolloutViolation) as caught:
            load_production_safety_policy(path, trust_store=trust_store, now=NOW)
        assert caught.value.code is RolloutCode.R_POLICY_UNAUTHORIZED
    uninitialized = object.__new__(ProductionApprovedSafetyPolicy)
    with pytest.raises(RolloutViolation):
        require_production_policy(uninitialized)


def test_policy_public_boundary_and_legacy_threshold_are_structural() -> None:
    parser = ast.parse((POLICY_SOURCE / "policy_parser.py").read_text(encoding="utf-8"))
    public_functions = {
        node.name
        for node in parser.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    assert public_functions == {
        "load_fixture_safety_policy",
        "load_production_safety_policy",
        "require_production_policy",
    }
    assert policy_canonical.__all__ == policy_io.__all__ == policy_schema.__all__ == ()
    assert policy_values.__all__ == ()
    assert policy_types.__all__ == (
        "FixtureApprovedSafetyPolicy",
        "ProductionApprovedSafetyPolicy",
    )
    for path in POLICY_SOURCE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if path.name != "policy_parser.py":
            assert "policy_values import" not in source
        assert "max_relative_target_degrees" not in source
    rollout_sources = "".join(
        path.read_text(encoding="utf-8") for path in POLICY_SOURCE.glob("rollout_*.py")
    )
    assert "dispatch_budget" not in rollout_sources
    assert "dispatch_budget_remaining" not in rollout_sources
