"""Todo 18 guarded single-step authorization and evidence contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real import single_step_fixture
from so101_pusht_benchmark.sim_to_real.ledger_chain import canonical_hash
from so101_pusht_benchmark.sim_to_real.physical_ik_scene_pose import scene_pose_content_digest
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode, RolloutViolation
from so101_pusht_benchmark.sim_to_real.single_step import (
    AcknowledgementEvidence,
    FixtureBus,
    PostStateEvidence,
    SingleStepBudget,
    SingleStepRunInput,
    SingleStepRuntime,
    run_fixture_single_step,
)

BENCHMARK = Path(__file__).resolve().parents[1]
FIXTURE = BENCHMARK / "tests/fixtures/sim_to_real/single_step_complete"
AUTHORIZATION = BENCHMARK / "tests/fixtures/sim_to_real/single_step_authorization.json"


class ThrowAfterWriteBus(FixtureBus):
    def sync_write(self, register: str, payload: dict[str, float]) -> None:
        super().sync_write(register, payload)
        raise RuntimeError("transport outcome unknown")


class AckProvider:
    def __init__(self, evidence: AcknowledgementEvidence) -> None:
        self.evidence = evidence

    def acknowledge(self) -> AcknowledgementEvidence:
        return self.evidence


class PostStateProvider:
    def __init__(self, evidence: PostStateEvidence) -> None:
        self.evidence = evidence

    def read_post_state(self) -> PostStateEvidence:
        return self.evidence


def _manifest() -> dict[str, object]:
    value: object = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _ack() -> AcknowledgementEvidence:
    raw = cast("dict[str, object]", _manifest()["acknowledgement"])
    return AcknowledgementEvidence.from_mapping(raw)


def _post() -> PostStateEvidence:
    raw = cast("dict[str, object]", _manifest()["post_state"])
    return PostStateEvidence.from_mapping(raw)


def _copy_authorization(tmp_path: Path, mutation: dict[str, object] | None = None) -> Path:
    target = tmp_path / "authorization.json"
    value: object = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    document = cast("dict[str, object]", value)
    if mutation is not None:
        document.update(mutation)
    target.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return target


@dataclass(frozen=True, slots=True)
class _RunOptions:
    authorization: Path = AUTHORIZATION
    fixture: Path = FIXTURE
    bus: FixtureBus | None = None
    ack: AcknowledgementEvidence | None = None
    post: PostStateEvidence | None = None
    budget: SingleStepBudget | None = None


def _run(
    tmp_path: Path,
    options: _RunOptions | None = None,
) -> tuple[dict[str, object], FixtureBus]:
    selected = options or _RunOptions()
    bus = selected.bus or FixtureBus()
    outcome = run_fixture_single_step(
        SingleStepRunInput(selected.fixture, selected.authorization, tmp_path),
        runtime=SingleStepRuntime(
            bus,
            AckProvider(selected.ack or _ack()),
            PostStateProvider(selected.post or _post()),
            selected.budget or SingleStepBudget(),
        ),
    )
    return outcome, bus


@pytest.mark.parametrize(
    ("field", "value"),
    [("sample_id", "other-single-step-sample"), ("sample_timestamp", 999.971)],
)
def test_rehashed_single_step_scene_pose_identity_drift_rejects(
    field: str,
    value: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = BENCHMARK / "tests/fixtures/sim_to_real/single_step_scene_pose.json"
    document = cast("dict[str, object]", json.loads(source.read_text(encoding="utf-8")))
    document[field] = value
    document["digest"] = scene_pose_content_digest(document)
    changed = tmp_path / "scene-pose.json"
    changed.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(single_step_fixture, "_SCENE_POSE", changed)

    with pytest.raises(RolloutViolation) as caught:
        single_step_fixture.physical_proposal()

    assert caught.value.code is RolloutCode.R_HASH_MISMATCH


def test_happy_write_count_is_one_with_provider_evidence(tmp_path: Path) -> None:
    outcome, bus = _run(tmp_path)

    assert outcome == {
        "acknowledgement_digest": _ack().digest,
        "command_id": "single-step-001",
        "evidence_scope": "test_fixture_only",
        "motor_writes_performed": True,
        "policy_evidence": "fixture_fake_bus_not_production",
        "post_state_digest": _post().digest,
        "state": "COMPLETE",
        "write_count": 1,
    }
    assert sum(entry[0] == "sync_write" for entry in bus.log) == 1
    assert json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8")) == outcome


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"expires_at": "2026-08-23T11:00:00Z"}, RolloutCode.R_HASH_MISMATCH),
        ({"policy_digest": "0" * 64}, RolloutCode.R_HASH_MISMATCH),
        ({"proposal_hash": "0" * 64}, RolloutCode.R_HASH_MISMATCH),
        ({"armed_receipt_digest": "0" * 64}, RolloutCode.R_HASH_MISMATCH),
        ({"command_id": "wrong-command"}, RolloutCode.R_HASH_MISMATCH),
    ],
    ids=("expired", "wrong-policy", "wrong-proposal", "wrong-armed-receipt", "wrong-command"),
)
def test_modified_authorization_never_dispatches(
    tmp_path: Path,
    mutation: dict[str, object],
    code: RolloutCode,
) -> None:
    bus = FixtureBus()
    authorization = _copy_authorization(tmp_path, mutation)

    with pytest.raises(RolloutViolation) as caught:
        _run(tmp_path, _RunOptions(authorization=authorization, bus=bus))

    assert caught.value.code is code
    assert bus.log == []


def test_arbitrary_existing_json_never_dispatches(tmp_path: Path) -> None:
    authorization = tmp_path / "authorization.json"
    authorization.write_text('{"token":"fixture-token"}', encoding="utf-8")
    bus = FixtureBus()

    with pytest.raises(RolloutViolation):
        _run(tmp_path, _RunOptions(authorization=authorization, bus=bus))

    assert bus.log == []


def test_authorization_has_one_call_budget(tmp_path: Path) -> None:
    budget = SingleStepBudget()
    _, bus = _run(tmp_path, _RunOptions(budget=budget))

    with pytest.raises(RolloutViolation) as caught:
        _run(tmp_path / "second", _RunOptions(bus=bus, budget=budget))

    assert caught.value.code is RolloutCode.R_DUPLICATE_DISPATCH
    assert sum(entry[0] == "sync_write" for entry in bus.log) == 1


def test_forged_ack_cannot_complete(tmp_path: Path) -> None:
    bus = FixtureBus()
    forged = replace(_ack(), accepted_body_degrees=(11.0, 20.0, 30.0, 40.0, 50.0))
    forged = replace(forged, digest=canonical_hash(forged.content()))

    with pytest.raises(RolloutViolation) as caught:
        _run(tmp_path, _RunOptions(bus=bus, ack=forged))

    assert caught.value.code is RolloutCode.R_ACK_MISMATCH
    assert not (tmp_path / "receipt.json").exists()
    assert sum(entry[0] == "sync_write" for entry in bus.log) == 1


def _stale_post(value: PostStateEvidence) -> PostStateEvidence:
    return replace(value, created_at=1000.01)


def _forged_post(value: PostStateEvidence) -> PostStateEvidence:
    return replace(value, digest="0" * 64)


def _echo_post(value: PostStateEvidence) -> PostStateEvidence:
    echoed = replace(value, sample_digest=_ack().provider_digest)
    return replace(echoed, digest=canonical_hash(echoed.content()))


PostMutation = Callable[[PostStateEvidence], PostStateEvidence]


@pytest.mark.parametrize(
    "post",
    [_stale_post, _forged_post, _echo_post],
    ids=("stale", "forged-digest", "fixture-echo"),
)
def test_stale_default_or_echo_post_state_cannot_complete(
    tmp_path: Path,
    post: PostMutation,
) -> None:
    bus = FixtureBus()
    mutated = post(_post())

    with pytest.raises(RolloutViolation):
        _run(tmp_path, _RunOptions(bus=bus, post=mutated))

    assert not (tmp_path / "receipt.json").exists()
    assert sum(entry[0] == "sync_write" for entry in bus.log) == 1


@pytest.mark.parametrize(
    ("filename", "mutation", "code"),
    [
        ("device_ownership.json", {"exclusive_owner": False}, RolloutCode.R_OWNERSHIP_CONFLICT),
        ("interlock.json", {"deadman_active": False}, RolloutCode.R_DEADMAN_INACTIVE),
        ("torque_state.json", {"state": "modified"}, RolloutCode.R_POLICY_UNAUTHORIZED),
        ("interlock.json", {"observed_at": "2026-08-23T11:00:00Z"}, RolloutCode.R_STALE),
        ("device_ownership.json", {"serial_device": "other-device"}, RolloutCode.R_HASH_MISMATCH),
    ],
    ids=("ownership", "deadman", "torque", "stale", "digest-binding"),
)
def test_independent_operational_evidence_blocks_before_writer_open(
    tmp_path: Path,
    filename: str,
    mutation: dict[str, object],
    code: RolloutCode,
) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE, fixture)
    path = fixture / "operational" / filename
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    document = cast("dict[str, object]", value)
    document.update(mutation)
    document["digest"] = canonical_hash(
        {key: item for key, item in document.items() if key != "digest"}
    )
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    bus = FixtureBus()

    with pytest.raises(RolloutViolation) as caught:
        _run(tmp_path / "out", _RunOptions(fixture=fixture, bus=bus))

    assert caught.value.code is code
    assert bus.log == []


def test_missing_operational_evidence_blocks_before_writer_open(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE, fixture)
    (fixture / "operational" / "torque_state.json").unlink()
    bus = FixtureBus()

    with pytest.raises(RolloutViolation) as caught:
        _run(tmp_path / "out", _RunOptions(fixture=fixture, bus=bus))

    assert caught.value.code is RolloutCode.R_MISSING
    assert bus.log == []


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["created_at", "body_degrees"])
def test_nonfinite_post_state_is_terminal_without_retry(
    tmp_path: Path,
    nonfinite: float,
    field: str,
) -> None:
    bus = FixtureBus()
    if field == "created_at":
        invalid = replace(_post(), created_at=nonfinite)
    else:
        invalid = replace(_post(), body_degrees=(nonfinite, 20.0, 30.0, 40.0, 50.0))

    with pytest.raises(RolloutViolation) as caught:
        _run(tmp_path, _RunOptions(bus=bus, post=invalid))

    assert caught.value.code is RolloutCode.F_POST_STATE_INVALID
    assert sum(entry[0] == "sync_write" for entry in bus.log) == 1
    assert not (tmp_path / "receipt.json").exists()


def test_throw_after_write_does_not_retry(tmp_path: Path) -> None:
    bus = ThrowAfterWriteBus()

    with pytest.raises(RolloutViolation) as caught:
        _run(tmp_path, _RunOptions(bus=bus))

    assert caught.value.code is RolloutCode.R_AMBIGUOUS_DISPATCH
    assert sum(entry[0] == "sync_write" for entry in bus.log) == 1


def test_missing_fixture_or_authorization_rejects_before_dispatch(tmp_path: Path) -> None:
    missing_fixture = tmp_path / "missing-fixture"
    bus = FixtureBus()
    with pytest.raises(RolloutViolation):
        run_fixture_single_step(
            SingleStepRunInput(missing_fixture, AUTHORIZATION, tmp_path),
            runtime=SingleStepRuntime(
                bus, AckProvider(_ack()), PostStateProvider(_post()), SingleStepBudget()
            ),
        )
    assert bus.log == []

    shutil.copytree(FIXTURE, tmp_path / "fixture")
    with pytest.raises(RolloutViolation):
        run_fixture_single_step(
            SingleStepRunInput(tmp_path / "fixture", tmp_path / "missing-auth", tmp_path),
            runtime=SingleStepRuntime(
                bus, AckProvider(_ack()), PostStateProvider(_post()), SingleStepBudget()
            ),
        )
    assert bus.log == []


def test_no_gripper_key_in_payload(tmp_path: Path) -> None:
    _, bus = _run(tmp_path)
    writes = [entry for entry in bus.log if entry[0] == "sync_write"]
    value = writes[0][1]
    assert isinstance(value, tuple)
    payload = value[1]
    assert "gripper" not in payload
    assert set(payload) == {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    }
