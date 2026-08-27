from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
import json
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real.joint_corpus_authority_cli import (
    assemble_authority,
    prepare_request,
)
from so101_pusht_benchmark.sim_to_real.joint_corpus_capture import (
    PoseRejectedError,
    RawReadBus,
    pose_check_message,
    pose_feedback,
    prove_manual_positioning_safe,
)
from so101_pusht_benchmark.sim_to_real.joint_corpus_capture_cli import (
    run_joint_corpus_capture_cli,
)
from so101_pusht_benchmark.sim_to_real.joint_corpus_contract import (
    TORQUE_NOT_PROVEN_INSTRUCTION,
)
from so101_pusht_benchmark.sim_to_real.joint_equivalence_corpus import (
    load_joint_corpus_documents,
    parse_joint_corpus,
)
from so101_pusht_benchmark.sim_to_real.joint_positioning_authority import (
    authority_document,
    load_joint_positioning_authority,
)
from so101_pusht_benchmark.sim_to_real.policy_approval import (
    ProductionTrustStore,
    RsaPkcs1v15Sha256Anchor,
)
from so101_pusht_benchmark.sim_to_real.policy_parser import load_fixture_safety_policy
from so101_pusht_benchmark.sim_to_real.read_only_authority import (
    ProductionReadOnlyAcquisitionAuthority,
    canonical_authority_bytes,
)
from so101_pusht_benchmark.sim_to_real.receipt_routing import CANONICAL_ROLLOUT_ROOT
from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutViolation
from so101_pusht_benchmark.sim_to_real.rsa_signing import (
    generate_rsa_private_key,
    public_key_from_private,
    rsa_pkcs1v15_sha256_sign,
)

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "tests/fixtures/sim_to_real/joint_corpus_capture_stream.json"
POLICY = ROOT / "tests/fixtures/sim_to_real/approved_policy.yaml"
SOURCE = ROOT / "src/so101_pusht_benchmark/sim_to_real/joint_corpus_capture.py"


def _args(session: Path, fixture: Path = STREAM) -> list[str]:
    return [
        "--fixture",
        str(fixture),
        "--policy",
        str(POLICY),
        "--session-dir",
        str(session),
        "--capture-id",
        "fixture-guided-capture",
    ]


def _mutated_stream(
    tmp_path: Path,
    mutate: Callable[[list[list[int]], dict[str, object]], object],
) -> Path:
    document = cast("dict[str, object]", json.loads(STREAM.read_text(encoding="utf-8")))
    mutate(cast("list[list[int]]", document["raw_encoder_samples"]), document)
    path = tmp_path / "stream.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_fixture_happy_path_builds_exact_unpublished_corpus(tmp_path: Path) -> None:
    session = tmp_path / "session"
    assert run_joint_corpus_capture_cli(_args(session)) == 0

    corpus, members = load_joint_corpus_documents(session)
    parsed = parse_joint_corpus(corpus, members, load_fixture_safety_policy(POLICY))
    assert len(parsed.members) == 15
    assert len(parsed.fit) == 13
    assert len(parsed.held_out) == 2
    assert corpus["publication_status"] == "owner_signature_required"
    assert corpus["genuine_scope_granted"] is False
    assert not (session / "corpus-authority.json").exists()
    for _, member in members:
        physical = cast("Mapping[str, object]", member["physical"])
        simulator = cast("Mapping[str, object]", member["simulator"])
        assert len(cast("list[object]", physical["raw_encoder_counts"])) == 5
        assert len(cast("list[object]", physical["joint_degrees"])) == 5
        assert len(cast("list[object]", simulator["joint_radians"])) == 5
        assert simulator["fk_oracle"] == "pinned_mujoco_model_recomputed_from_raw_vectors"


def _duplicate(samples: list[list[int]], _document: dict[str, object]) -> None:
    samples[1] = list(samples[0])


def _insufficient_span(samples: list[list[int]], _document: dict[str, object]) -> None:
    samples[1][0] = samples[0][0] - 20


def _wrong_sign(samples: list[list[int]], _document: dict[str, object]) -> None:
    samples[1] = list(samples[6])


@pytest.mark.parametrize(
    "mutation",
    [_duplicate, _insufficient_span, _wrong_sign],
    ids=["duplicate-static", "insufficient-span", "wrong-sign"],
)
def test_bad_pose_streams_never_publish_complete_corpus(
    tmp_path: Path,
    mutation: Callable[[list[list[int]], dict[str, object]], object],
) -> None:
    fixture = _mutated_stream(tmp_path, mutation)
    session = tmp_path / "session"
    assert run_joint_corpus_capture_cli(_args(session, fixture)) == 2
    assert not (session / "corpus.json").exists()


def test_interrupted_session_is_partial_then_resumes_exact_next_pose(tmp_path: Path) -> None:
    session = tmp_path / "session"
    assert run_joint_corpus_capture_cli([*_args(session), "--fixture-stop-after", "4"]) == 3
    assert len(list((session / "members").glob("*.json"))) == 4
    assert not (session / "corpus.json").exists()

    assert run_joint_corpus_capture_cli(_args(session)) == 0
    assert len(list((session / "members").glob("*.json"))) == 15
    assert (session / "corpus.json").is_file()


def test_identity_drift_and_member_tamper_block_resume(tmp_path: Path) -> None:
    fixture = tmp_path / "stream.json"
    shutil.copyfile(STREAM, fixture)
    session = tmp_path / "identity-session"
    assert (
        run_joint_corpus_capture_cli([*_args(session, fixture), "--fixture-stop-after", "1"]) == 3
    )
    fixture.write_text(fixture.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert run_joint_corpus_capture_cli(_args(session, fixture)) == 2
    assert not (session / "corpus.json").exists()

    tamper_session = tmp_path / "tamper-session"
    assert run_joint_corpus_capture_cli([*_args(tamper_session), "--fixture-stop-after", "1"]) == 3
    member = tamper_session / "members/fit-baseline.json"
    member.write_text(member.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert run_joint_corpus_capture_cli(_args(tamper_session)) == 2
    assert not (tamper_session / "corpus.json").exists()


def test_torque_not_proven_has_exact_instruction_and_zero_bus_calls() -> None:
    class Bus:
        calls = 0

        def sync_read(self, *_args: object, **_kwargs: object) -> Mapping[str, int]:
            self.calls += 1
            return {}

    bus = Bus()
    with pytest.raises(RolloutViolation) as caught:
        prove_manual_positioning_safe(cast("RawReadBus", bus), ())
    assert str(caught.value) == f"R_POLICY_UNAUTHORIZED: {TORQUE_NOT_PROVEN_INSTRUCTION}"
    assert bus.calls == 0


def _enable_elbow_torque(_samples: list[list[int]], document: dict[str, object]) -> None:
    document["torque_enabled"] = [0, 0, 1, 0, 0, 0]


def test_enabled_torque_preflight_and_symlink_route_fail_without_corpus(tmp_path: Path) -> None:
    fixture = _mutated_stream(tmp_path, _enable_elbow_torque)
    session = tmp_path / "torque-session"
    assert run_joint_corpus_capture_cli([*_args(session, fixture), "--preflight-only"]) == 2
    assert not (session / "corpus.json").exists()

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "linked-session"
    symlink.symlink_to(target, target_is_directory=True)
    assert run_joint_corpus_capture_cli(_args(symlink)) == 2
    assert not (target / "corpus.json").exists()


def test_signed_positioning_authority_binds_base_identity_and_permissions(
    tmp_path: Path,
) -> None:
    private = generate_rsa_private_key()
    public = public_key_from_private(private)
    signer = hashlib.sha256(public).hexdigest()
    base = cast(
        "ProductionReadOnlyAcquisitionAuthority",
        cast(
            "object",
            SimpleNamespace(
                canonical_digest="a" * 64,
                source_lineage_authority_digest="b" * 64,
                follower_device_digest="c" * 64,
                calibration_digest="d" * 64,
            ),
        ),
    )
    now = datetime(2026, 8, 25, 12, tzinfo=timezone.utc)
    document = authority_document(
        base,
        authority_id="positioning-test",
        approved_by=signer,
        valid_from=now,
    )
    encoded = canonical_authority_bytes(document)
    authority = tmp_path / "positioning.json"
    signature = tmp_path / "positioning.sig"
    authority.write_bytes(encoded)
    signature.write_bytes(rsa_pkcs1v15_sha256_sign(private, encoded))
    trust = ProductionTrustStore.from_owner_anchors((RsaPkcs1v15Sha256Anchor(public),))

    loaded = load_joint_positioning_authority(
        authority,
        signature_path=signature,
        trust_store=trust,
        base=base,
        now=now,
    )
    assert loaded.permissions[1] == "sync_read:Torque_Enable"
    assert loaded.acquisition_authority_digest == "a" * 64
    signature.write_bytes(b"tampered")
    with pytest.raises(RolloutViolation, match="untrusted"):
        load_joint_positioning_authority(
            authority,
            signature_path=signature,
            trust_store=trust,
            base=base,
            now=now,
        )


def test_exact_member_authority_requires_verified_owner_signature(tmp_path: Path) -> None:
    session = tmp_path / "session"
    assert run_joint_corpus_capture_cli(_args(session)) == 0
    private = generate_rsa_private_key()
    public = public_key_from_private(private)
    anchor = tmp_path / "owner.pem"
    anchor.write_bytes(public)
    with TemporaryDirectory(prefix="pytest-joint-authority-", dir=CANONICAL_ROLLOUT_ROOT) as raw:
        output_dir = Path(raw)
        prepared = prepare_request(session, anchor, "owner-approval-001", output_dir)
        binding = Path(cast("str", prepared["binding_path"]))
        signature = output_dir / "corpus-authority-binding.sig"
        signature.write_bytes(rsa_pkcs1v15_sha256_sign(private, binding.read_bytes()))
        authority = output_dir / "corpus-authority.json"
        result = assemble_authority(
            Path(cast("str", prepared["request_path"])),
            binding,
            signature,
            anchor,
            authority,
        )
        assert result["owner_signature_verified"] is True
        assert result["exact_member_count"] == 15
        assert authority.is_file()

        bad_signature = output_dir / "bad.sig"
        bad_signature.write_bytes(b"not a signature")
        rejected = output_dir / "must-not-publish.json"
        with pytest.raises(ValueError, match="untrusted"):
            assemble_authority(
                Path(cast("str", prepared["request_path"])),
                binding,
                bad_signature,
                anchor,
                rejected,
            )
        assert not rejected.exists()


def test_capture_source_exposes_no_hardware_write_or_timing_sleep() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
    assert attributes.isdisjoint({"sync_write", "write", "sleep", "configure", "calibrate"})
    source = SOURCE.read_text(encoding="utf-8")
    assert "Goal_Position" not in source
    assert "disable_torque=True" not in source


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        ("isolated pose has wrong sign", "반대 방향"),
        ("isolated pose has insufficient span", "조금 더"),
        ("isolated pose changed another joint too far", "다른 관절"),
        ("pose is duplicate/static; reposition", "거의 움직이지"),
        ("combination pose needs sufficient span", "두 관절 이상"),
    ],
)
def test_pose_feedback_uses_plain_physical_corrections(detail: str, expected: str) -> None:
    assert expected in pose_feedback(PoseRejectedError(detail))


def test_pose_check_message_names_joint_that_must_return() -> None:
    baseline = {"physical": {"joint_degrees": [0.0, 0.0, 0.0, 0.0, 0.0]}}
    pose = next(
        item
        for item in __import__(
            "so101_pusht_benchmark.sim_to_real.joint_corpus_contract",
            fromlist=["POSE_PLAN"],
        ).POSE_PLAN
        if item.identifier == "fit-shoulder_pan-neg"
    )
    message = pose_check_message(pose, [-10.0, 8.0, 0.0, 0.0, 0.0], [baseline])
    assert "shoulder_lift" in message
    assert "+8.0°" in message


def test_operator_confirmation_accepts_lowercase_commands() -> None:
    from so101_pusht_benchmark.sim_to_real.joint_corpus_capture_cli import (
        normalize_confirmation,
    )

    assert normalize_confirmation("check") == "CHECK"
    assert normalize_confirmation("capture fit-baseline") == "CAPTURE fit-baseline"
    assert normalize_confirmation("stop") == "STOP"


def test_out_of_range_check_gives_recovery_instruction() -> None:
    from so101_pusht_benchmark.sim_to_real.joint_corpus_capture import (
        check_violation_feedback,
    )
    from so101_pusht_benchmark.sim_to_real.rollout_codes import RolloutCode

    violation = RolloutViolation(RolloutCode.R_OUT_OF_RANGE, "wrist_flex degree 100.35")
    assert check_violation_feedback(violation) == (
        "허용 범위를 넘었습니다. 같은 방향으로 더 움직이지 말고 반대쪽으로 되돌린 뒤 다시 check 하세요. "
        "wrist_flex degree 100.35"
    )


def test_operator_confirmation_accepts_single_key_commands() -> None:
    from so101_pusht_benchmark.sim_to_real.joint_corpus_capture_cli import (
        normalize_confirmation,
    )

    assert normalize_confirmation("", "fit-baseline") == "CAPTURE fit-baseline"
    assert normalize_confirmation("c", "fit-baseline") == "CHECK"
    assert normalize_confirmation("s", "fit-baseline") == "CAPTURE fit-baseline"
    assert normalize_confirmation("q", "fit-baseline") == "STOP"


def test_keyboard_confirmation_enter_captures_current_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    from so101_pusht_benchmark.sim_to_real.joint_corpus_capture_cli import (
        KeyboardConfirmation,
    )
    from so101_pusht_benchmark.sim_to_real.joint_corpus_contract import PoseInstruction

    def empty_input(_prompt: str) -> str:
        return ""

    monkeypatch.setattr("builtins.input", empty_input)
    pose = PoseInstruction("fit-baseline", "fit", "baseline", "neutral")
    assert KeyboardConfirmation()(pose) == "CAPTURE fit-baseline"


def test_korean_progress_message_explains_saved_and_next_pose() -> None:
    from so101_pusht_benchmark.sim_to_real.joint_corpus_capture import korean_progress_message

    assert korean_progress_message(1) == (
        "[저장 완료 1/15] 기준 자세를 저장했습니다.\n"
        "[다음 2/15] 맨 아래 베이스 회전축(shoulder_pan) 하나만 아무 한쪽으로 5~10도 움직이세요."
    )


def test_korean_rejection_message_explains_problem_and_retry() -> None:
    from so101_pusht_benchmark.sim_to_real.joint_corpus_capture import (
        korean_rejection_message,
    )

    assert korean_rejection_message(2, "elbow_flex를 기준 자세로 되돌리세요") == (
        "[저장 안 됨 2/15] 현재 자세가 조건에 맞지 않습니다.\n"
        "[문제와 수정 방법] elbow_flex를 기준 자세로 되돌리세요\n"
        "[다시 시도] 자세를 고치고 손을 완전히 뗀 뒤 Enter를 누르세요."
    )
