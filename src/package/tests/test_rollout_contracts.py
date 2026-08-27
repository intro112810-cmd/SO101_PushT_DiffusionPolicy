from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import inspect

import pytest

from so101_pusht_benchmark.sim_to_real import (
    rollout_authority,
    rollout_codes,
    rollout_identity,
    rollout_record_types,
    rollout_records,
    rollout_snapshot,
    rollout_state_machine,
)
from so101_pusht_benchmark.sim_to_real.contracts import (
    MOTORS,
    ContractError,
    validate_follower_receipt,
)
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.rollout_record_types import RolloutRecordVariant
from so101_pusht_benchmark.sim_to_real.rollout_records import (
    BoundaryValue,
    digest_content,
    parse_record,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
BODY = [0.0, -10.0, 20.0, 5.0, 0.0]


def _sealed(kind: str, **values: BoundaryValue) -> dict[str, BoundaryValue]:
    content: dict[str, BoundaryValue] = {"kind": kind, **values}
    content["digest"] = digest_content(content)
    return content


def _parsed(kind: str, **values: BoundaryValue) -> RolloutRecordVariant:
    return parse_record(_sealed(kind, **values))


def _chain() -> tuple[RolloutRecordVariant, ...]:
    sample = _parsed(
        "physical_sample",
        record_id="sample-1",
        created_at=100.0,
        camera_timestamp=99.9,
        joint_timestamp=99.95,
        frame_digest=SHA_A,
        body_degrees=BODY,
        device_digest=SHA_A,
        calibration_digest=SHA_B,
    )
    proposal = _parsed(
        "proposal",
        record_id="proposal-1",
        created_at=100.1,
        sample_digest=sample.digest,
        target_xy=[0.1, 0.2],
        policy_digest=SHA_A,
    )
    evidence = _parsed(
        "evidence",
        record_id="evidence-1",
        created_at=100.2,
        proposal_digest=proposal.digest,
        evidence_type="joint_and_camera_equivalence",
        artifact_digest=SHA_B,
        valid_until=102.0,
    )
    authorization = _parsed(
        "authorization",
        record_id="authorization-1",
        created_at=100.3,
        proposal_digest=proposal.digest,
        evidence_digest=evidence.digest,
        policy_digest=SHA_A,
        valid_until=101.0,
    )
    command = _parsed(
        "command",
        record_id="command-1",
        created_at=100.4,
        proposal_digest=proposal.digest,
        authorization_digest=authorization.digest,
        body_degrees=BODY,
    )
    acknowledgement = _parsed(
        "acknowledgement",
        record_id="ack-1",
        created_at=100.5,
        command_digest=command.digest,
        provider_digest=SHA_B,
        accepted_body_degrees=BODY,
    )
    post_state = _parsed(
        "post_state",
        record_id="post-1",
        created_at=100.6,
        command_digest=command.digest,
        acknowledgement_digest=acknowledgement.digest,
        sample_digest=SHA_A,
        body_degrees=BODY,
    )
    return sample, proposal, evidence, authorization, command, acknowledgement, post_state


def test_existing_follower_contract_characterization() -> None:
    receipt = {
        "mode": "read_only_follower_state",
        "motor_writes_performed": False,
        "actuation_performed": False,
        "positions_degrees": {motor: float(index) for index, motor in enumerate(MOTORS)},
    }

    assert validate_follower_receipt(receipt) == {
        motor: float(index) for index, motor in enumerate(MOTORS)
    }
    with pytest.raises(ContractError, match="exactly six motors"):
        validate_follower_receipt({**receipt, "positions_degrees": {"elbow_flex": 1.0}})


def test_complete_cycle_contract() -> None:
    records = _chain()

    assert len({record.digest for record in records}) == len(records)
    assert all(len(record.digest) == 64 for record in records)
    assert _chain() == records
    with pytest.raises(FrozenInstanceError):
        records[0].__setattr__("record_id", "mutated")


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"created_at": None}, RolloutCode.R_MISSING),
        ({"created_at": float("nan")}, RolloutCode.R_NONFINITE),
        ({"created_at": 90.0}, RolloutCode.R_STALE),
        ({"digest": SHA_B}, RolloutCode.R_HASH_MISMATCH),
    ],
)
def test_missing_nonfinite_stale_hash_mismatch_codes(
    mutation: dict[str, BoundaryValue], code: RolloutCode
) -> None:
    raw = _sealed(
        "physical_sample",
        record_id="sample-1",
        created_at=100.0,
        camera_timestamp=99.9,
        joint_timestamp=99.95,
        frame_digest=SHA_A,
        body_degrees=BODY,
        device_digest=SHA_A,
        calibration_digest=SHA_B,
    )
    raw.update(mutation)
    if "digest" not in mutation:
        raw["digest"] = digest_content(
            {key: value for key, value in raw.items() if key != "digest"}
        )

    with pytest.raises(RolloutViolation) as caught:
        parse_record(raw, now=100.0, max_age=1.0)
    assert caught.value.code is code


def test_plan_derived_sentinel_code_contract() -> None:
    plan_sentinels = (
        "EQUIVALENCE_UNPROVEN",
        "CAMERA_UNREGISTERED",
        "HISTORY_INCOMPLETE",
        "R_INVALID_ELBOW",
        "R_POLICY_UNAUTHORIZED",
        "R_CLIPPING_REQUIRED",
    )

    vocabulary = {code.name: code.value for code in RolloutCode}
    assert all(vocabulary[sentinel] == sentinel for sentinel in plan_sentinels)
    assert all(code.name == code.value for code in RolloutCode)


def test_typed_numeric_spellings_have_one_canonical_digest() -> None:
    integer_spelling = _sealed(
        "physical_sample",
        record_id="sample-canonical",
        created_at=30,
        camera_timestamp=29,
        joint_timestamp=29,
        frame_digest=SHA_A,
        body_degrees=[0, -10, 20, 5, 0],
        device_digest=SHA_A,
        calibration_digest=SHA_B,
    )
    float_spelling = _sealed(
        "physical_sample",
        record_id="sample-canonical",
        created_at=30.0,
        camera_timestamp=29.0,
        joint_timestamp=29.0,
        frame_digest=SHA_A,
        body_degrees=[0.0, -10.0, 20.0, 5.0, 0.0],
        device_digest=SHA_A,
        calibration_digest=SHA_B,
    )

    integer_record = parse_record(integer_spelling)
    float_record = parse_record(float_spelling)
    assert integer_record == float_record
    assert integer_record.digest == float_record.digest


def test_todo2_public_contract_has_no_writer_authorization_surface() -> None:
    modules = (
        rollout_authority,
        rollout_codes,
        rollout_identity,
        rollout_record_types,
        rollout_records,
        rollout_snapshot,
        rollout_state_machine,
    )
    forbidden_names = {
        "authorize_write",
        "authorization_token",
        "supervisor_token",
        "writer_capability",
        "writer_token",
    }
    forbidden_imports = {"dynamixel_sdk", "lerobot", "serial"}
    forbidden_calls = forbidden_names | {
        "open_device",
        "send_action",
        "sync_write",
        "write_goal_position",
    }

    for module in modules:
        assert all(hasattr(module, name) for name in module.__all__)
        assert forbidden_names.isdisjoint(module.__all__)
        tree = ast.parse(inspect.getsource(module))
        declared = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        assert forbidden_names.isdisjoint(declared)
        assert forbidden_imports.isdisjoint(imported)
        assert forbidden_calls.isdisjoint(called)


def test_future_physical_authority_requires_todo13_supervisor_token() -> None:
    assert rollout_snapshot.TODO2_PHYSICAL_AUTHORITY_BOUNDARY == (
        "TODO2_STATE_ONLY__TODO13_SUPERVISOR_TOKEN_REQUIRED__TODO16_CONSUMES"
    )
    assert "RolloutSnapshot" not in rollout_snapshot.TODO2_PHYSICAL_AUTHORITY_BOUNDARY
    assert "Authorization" not in rollout_snapshot.TODO2_PHYSICAL_AUTHORITY_BOUNDARY
