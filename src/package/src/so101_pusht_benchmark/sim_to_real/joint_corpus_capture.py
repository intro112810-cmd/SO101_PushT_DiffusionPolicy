"""Resumable, no-actuation guided capture of a genuine joint/FK corpus candidate."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import cast, Protocol

from .joint_corpus_contract import (
    JOINT_CORPUS_PROVIDER_DIGEST,
    MAX_OTHER_JOINT_DRIFT_DEGREES,
    MIN_DISTINCT_POSE_DELTA_DEGREES,
    MIN_ISOLATED_DELTA_DEGREES,
    POSE_PLAN,
    TORQUE_NOT_PROVEN_INSTRUCTION,
    TORQUE_PERMISSION,
    TORQUE_RESISTING_INSTRUCTION,
    PoseInstruction,
    pose_plan_digest,
)
from .joint_equivalence_corpus import CORPUS_SCHEMA, MEMBER_SCHEMA, canonical_digest
from .joint_equivalence_fk import derive_pinned_fk_positions
from .joint_mapping import ENCODER_MAX, JOINT_ORDER
from .physical_ik_fk import build_joint_domains, degrees_to_radians, parse_body_degrees
from .policy_types import FixtureApprovedSafetyPolicy, ProductionApprovedSafetyPolicy
from .read_only_authority import ProductionReadOnlyAcquisitionAuthority
from .read_only_authority_io import path_metadata_digest
from .receipt_routing import (
    locate_receipt_path,
    prepare_receipt_directory,
    validate_receipt_identity,
)
from .rollout_codes import RolloutCode, RolloutViolation
from .secure_io import atomic_write_new, read_regular_leaf

CapturePolicy = FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy
Clock = Callable[[], float]
WallClock = Callable[[], datetime]


class RawReadBus(Protocol):
    """Only the direct-bus methods required by the capture."""

    def connect(self) -> None: ...

    def sync_read(
        self,
        register: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
    ) -> Mapping[str, int | float]: ...

    def disconnect(self, *, disable_torque: bool) -> None: ...


class ConfirmationSource(Protocol):
    def __call__(self, pose: PoseInstruction) -> str: ...


@dataclass(frozen=True, slots=True)
class CaptureIdentity:
    """Exact authority/device/calibration identity frozen for one session."""

    authority_digest: str
    policy_digest: str
    provider_digest: str
    device_digest: str
    calibration_digest: str
    capture_id: str
    profile_path: Path
    profile_digest: str
    calibration_path: Path
    follower_path: Path
    production: bool


@dataclass(frozen=True, slots=True)
class EncoderCalibration:
    zero_counts: tuple[float, float, float, float, float]
    degrees_per_count: tuple[float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class CaptureOutcome:
    status: str
    captured_count: int
    required_count: int
    next_pose: str | None
    corpus_path: Path | None
    session_identity_digest: str


@dataclass(frozen=True, slots=True)
class MemberCaptureInput:
    pose: PoseInstruction
    raw_counts: tuple[float, float, float, float, float]
    calibration: EncoderCalibration
    policy: CapturePolicy
    identity: CaptureIdentity
    confirmation: str
    confirmed_at: datetime
    read_started: float
    read_completed: float
    derived_at: float


@dataclass(frozen=True, slots=True)
class GuidedCaptureRequest:
    session_root: Path
    identity: CaptureIdentity
    policy: CapturePolicy | None
    bus: RawReadBus
    confirm: ConfirmationSource
    clock: Clock
    wall_clock: WallClock
    follower_permissions: Sequence[str]
    preflight_only: bool = False


class CaptureStoppedError(RuntimeError):
    """Operator explicitly stopped after preserving all completed members."""


class PoseRejectedError(ValueError):
    """A confirmed read is not sufficient for its declared inventory slot."""


def canonical_json(value: Mapping[str, object]) -> bytes:
    """Canonical bytes used for all immutable session leaves."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"unsafe identity file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_document(identity: CaptureIdentity) -> dict[str, object]:
    return {
        "schema": "so101-guided-joint-corpus-session-v1",
        "authority_digest": identity.authority_digest,
        "policy_digest": identity.policy_digest,
        "provider_digest": identity.provider_digest,
        "device_digest": identity.device_digest,
        "calibration_digest": identity.calibration_digest,
        "capture_id": identity.capture_id,
        "profile_path": str(identity.profile_path),
        "profile_digest": identity.profile_digest,
        "calibration_path": str(identity.calibration_path),
        "follower_path": str(identity.follower_path),
        "production": identity.production,
        "pose_plan_digest": pose_plan_digest(),
        "required_member_count": len(POSE_PLAN),
        "evidence_scope": "unpublished_physical_capture_candidate",
        "motor_writes_performed": False,
        "actuation_performed": False,
    }


def _calibration(path: Path) -> EncoderCalibration:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "calibration is unreadable") from exc
    if not isinstance(raw, Mapping):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "calibration must be a mapping")
    entries = cast("Mapping[str, object]", raw)
    zeros: list[float] = []
    scales: list[float] = []
    ids: list[int] = []
    for joint in JOINT_ORDER:
        value = entries.get(joint)
        if not isinstance(value, Mapping):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"calibration lacks {joint}")
        item = cast("Mapping[str, object]", value)
        if not all(isinstance(item.get(key), int) for key in ("id", "range_min", "range_max")):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"calibration {joint} is invalid")
        identifier = cast("int", item["id"])
        minimum = cast("int", item["range_min"])
        maximum = cast("int", item["range_max"])
        if minimum >= maximum:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, f"calibration {joint} range")
        ids.append(identifier)
        zeros.append((minimum + maximum) / 2)
        scales.append(360.0 / ENCODER_MAX)
    if ids != [1, 2, 3, 4, 5]:
        raise RolloutViolation(RolloutCode.R_PROVIDER_MISMATCH, "calibration motor order drift")
    return EncoderCalibration(
        cast("tuple[float, float, float, float, float]", tuple(zeros)),
        cast("tuple[float, float, float, float, float]", tuple(scales)),
    )


def _read_vector(
    values: Mapping[str, int | float], label: str
) -> tuple[float, float, float, float, float]:
    expected = frozenset((*JOINT_ORDER, "gripper"))
    if frozenset(values) != expected:
        raise RolloutViolation(RolloutCode.R_PROVIDER_MISMATCH, f"{label} motor set drift")
    vector = tuple(float(values[joint]) for joint in JOINT_ORDER)
    if not all(math.isfinite(value) for value in vector):
        raise RolloutViolation(RolloutCode.R_NONFINITE, label)
    if label == "Present_Position" and any(not value.is_integer() for value in vector):
        raise RolloutViolation(RolloutCode.R_PROVIDER_MISMATCH, "raw encoder count is non-integral")
    return cast("tuple[float, float, float, float, float]", vector)


def prove_manual_positioning_safe(bus: RawReadBus, permissions: Sequence[str]) -> dict[str, int]:
    """Prove every motor has torque disabled without ever changing torque state."""
    if TORQUE_PERMISSION not in permissions:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, TORQUE_NOT_PROVEN_INSTRUCTION)
    try:
        observed = bus.sync_read("Torque_Enable", normalize=False)
        vector = _read_vector(observed, "torque state")
    except Exception as exc:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, TORQUE_NOT_PROVEN_INSTRUCTION
        ) from exc
    if any(value != 0.0 for value in vector) or float(observed["gripper"]) != 0.0:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, TORQUE_RESISTING_INSTRUCTION)
    return {name: int(observed[name]) for name in (*JOINT_ORDER, "gripper")}


def _pose_deltas(
    degrees: Sequence[float], baseline: Sequence[float]
) -> tuple[float, float, float, float, float]:
    return cast(
        "tuple[float, float, float, float, float]",
        tuple(value - base for value, base in zip(degrees, baseline, strict=True)),
    )


def validate_pose(
    pose: PoseInstruction,
    degrees: Sequence[float],
    prior: Sequence[Mapping[str, object]],
) -> None:
    """Reject static, wrong-sign, drifting, or insufficient inventory members."""
    for member in prior:
        physical = cast("Mapping[str, object]", member["physical"])
        previous = cast("Sequence[float]", physical["joint_degrees"])
        if max(abs(a - b) for a, b in zip(degrees, previous, strict=True)) < (
            MIN_DISTINCT_POSE_DELTA_DEGREES
        ):
            raise PoseRejectedError("pose is duplicate/static; reposition before confirming")
    if pose.category == "baseline":
        return
    baseline_physical = cast("Mapping[str, object]", prior[0]["physical"])
    baseline = cast("Sequence[float]", baseline_physical["joint_degrees"])
    deltas = _pose_deltas(degrees, baseline)
    if pose.category == "isolated":
        if pose.isolated_joint is None:
            raise PoseRejectedError("isolated pose contract is invalid")
        target = JOINT_ORDER.index(pose.isolated_joint)
        directed = deltas[target] * pose.direction
        if directed <= 0:
            raise PoseRejectedError("isolated pose has wrong sign")
        if directed < MIN_ISOLATED_DELTA_DEGREES:
            raise PoseRejectedError("isolated pose has insufficient span")
        if any(
            abs(delta) > MAX_OTHER_JOINT_DRIFT_DEGREES
            for index, delta in enumerate(deltas)
            if index != target
        ):
            raise PoseRejectedError("isolated pose changed another joint too far")
        return
    changed = sum(abs(delta) >= MIN_ISOLATED_DELTA_DEGREES for delta in deltas)
    if changed < 2:
        raise PoseRejectedError("combination pose needs sufficient span on at least two joints")


def _member(inputs: MemberCaptureInput) -> dict[str, object]:
    pose = inputs.pose
    identity = inputs.identity
    degrees = parse_body_degrees(
        tuple(
            (count - zero) * scale
            for count, zero, scale in zip(
                inputs.raw_counts,
                inputs.calibration.zero_counts,
                inputs.calibration.degrees_per_count,
                strict=True,
            )
        )
    )
    radians = degrees_to_radians(degrees, build_joint_domains(inputs.policy))
    xyz = derive_pinned_fk_positions((radians,))[0]
    timestamp = (inputs.read_started + inputs.read_completed) / 2
    return {
        "schema": MEMBER_SCHEMA,
        "sample_id": pose.identifier,
        "physical": {
            "timestamp_s": timestamp,
            "read_started_at_monotonic_s": inputs.read_started,
            "read_completed_at_monotonic_s": inputs.read_completed,
            "captured_at_utc": inputs.confirmed_at.astimezone(timezone.utc).isoformat(),
            "raw_encoder_counts": [int(value) for value in inputs.raw_counts],
            "joint_degrees": list(degrees),
            "measured_tool_xyz_m": list(xyz),
            "device_digest": identity.device_digest,
            "calibration_digest": identity.calibration_digest,
        },
        "simulator": {
            "timestamp_s": timestamp,
            "derived_at_monotonic_s": inputs.derived_at,
            "joint_order": list(JOINT_ORDER),
            "joint_radians": list(radians),
            "tool_xyz_m": list(xyz),
            "fk_oracle": "pinned_mujoco_model_recomputed_from_raw_vectors",
        },
        "operator_confirmation": {
            "event": inputs.confirmation,
            "pose_id": pose.identifier,
            "confirmed_at_utc": inputs.confirmed_at.astimezone(timezone.utc).isoformat(),
        },
        "capture_identity": {
            "authority_digest": identity.authority_digest,
            "provider_digest": identity.provider_digest,
            "device_digest": identity.device_digest,
            "calibration_digest": identity.calibration_digest,
            "capture_id": identity.capture_id,
        },
    }


def _publish(path: Path, document: Mapping[str, object]) -> None:
    content = canonical_json(document)
    location = locate_receipt_path(path)
    atomic_write_new(
        location.resolved.parent,
        location.resolved.name,
        content,
        temporary=f".{path.name}.capture-{os.getpid()}.tmp",
    )
    persisted, _ = read_regular_leaf(location.resolved.parent, location.resolved.name)
    if persisted != content:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "capture publication drift")


def _load_member(path: Path) -> Mapping[str, object]:
    content, _ = read_regular_leaf(path.parent.resolve(strict=True), path.name)
    try:
        value: object = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "captured member is invalid") from exc
    if (
        not isinstance(value, Mapping)
        or canonical_json(cast("Mapping[str, object]", value)) != content
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "captured member is noncanonical")
    return cast("Mapping[str, object]", value)


def _prepare_session(root: Path, identity: CaptureIdentity) -> tuple[Path, str]:
    validate_receipt_identity(locate_receipt_path(root), production=identity.production)
    prepare_receipt_directory(root, production=identity.production)
    members = root / "members"
    prepare_receipt_directory(members, production=identity.production)
    expected = _identity_document(identity)
    digest = hashlib.sha256(canonical_json(expected)).hexdigest()
    identity_path = root / "session-identity.json"
    if identity_path.exists():
        if _load_member(identity_path) != expected:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "resumed session identity drift")
    else:
        _publish(identity_path, expected)
    return members, digest


def _existing_members(root: Path) -> list[Mapping[str, object]]:
    members: list[Mapping[str, object]] = []
    gap = False
    for pose in POSE_PLAN:
        path = root / "members" / f"{pose.identifier}.json"
        if not path.exists():
            gap = True
            continue
        if gap:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "captured pose order drift")
        document = _load_member(path)
        if document.get("sample_id") != pose.identifier:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "captured pose identity drift")
        physical = cast("Mapping[str, object]", document.get("physical"))
        degrees = cast("Sequence[float]", physical.get("joint_degrees"))
        validate_pose(pose, degrees, members)
        members.append(document)
    return members


def _verify_live_identity(identity: CaptureIdentity) -> None:
    if _sha256_file(identity.profile_path) != identity.profile_digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "profile identity drift")
    if _sha256_file(identity.calibration_path) != identity.calibration_digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "calibration identity drift")
    if path_metadata_digest(identity.follower_path) != identity.device_digest:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "follower device identity drift")


def _manifest(
    members: Sequence[Mapping[str, object]],
    calibration: EncoderCalibration,
    policy: CapturePolicy,
    identity: CaptureIdentity,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for pose, member in zip(POSE_PLAN, members, strict=True):
        entry: dict[str, object] = {
            "id": pose.identifier,
            "sample_id": pose.identifier,
            "path": f"members/{pose.identifier}.json",
            "sha256": hashlib.sha256(canonical_json(member)).hexdigest(),
            "split": pose.split,
            "category": pose.category,
        }
        if pose.isolated_joint is not None:
            entry["isolated_joint"] = pose.isolated_joint
        entries.append(entry)
    manifest: dict[str, object] = {
        "schema": CORPUS_SCHEMA,
        "evidence_origin": "physical_read_only_capture",
        "simulator_joint_order": list(JOINT_ORDER),
        "encoder_calibration": {
            "zero_counts": list(calibration.zero_counts),
            "degrees_per_count": list(calibration.degrees_per_count),
        },
        "members": entries,
        "policy_digest": policy.canonical_digest,
        "production_bindings": {
            "provider_digest": identity.provider_digest,
            "device_digest": identity.device_digest,
            "calibration_digest": identity.calibration_digest,
            "capture_id": identity.capture_id,
        },
        "publication_status": "owner_signature_required",
        "genuine_scope_granted": False,
    }
    manifest["corpus_digest"] = canonical_digest(manifest)
    return manifest


def pose_check_message(
    pose: PoseInstruction,
    degrees: Sequence[float],
    prior: Sequence[Mapping[str, object]],
) -> str:
    """Explain the next physical correction without sign terminology."""
    if pose.category == "baseline":
        return "OK: 이 자세가 기준 자세로 저장됩니다."
    baseline_physical = cast("Mapping[str, object]", prior[0]["physical"])
    baseline = cast("Sequence[float]", baseline_physical["joint_degrees"])
    deltas = _pose_deltas(degrees, baseline)
    if pose.category == "isolated" and pose.isolated_joint is not None:
        target = JOINT_ORDER.index(pose.isolated_joint)
        drift = [
            (JOINT_ORDER[index], delta)
            for index, delta in enumerate(deltas)
            if index != target and abs(delta) > MAX_OTHER_JOINT_DRIFT_DEGREES
        ]
        if drift:
            detail = ", ".join(f"{name} {delta:+.1f}°" for name, delta in drift)
            return f"다른 관절을 기준 자세로 되돌리세요: {detail}"
        directed = deltas[target] * pose.direction
        if directed <= 0:
            return f"{pose.isolated_joint}: 지금 움직인 방향의 정확히 반대쪽으로 움직이세요."
        if directed < MIN_ISOLATED_DELTA_DEGREES:
            return f"{pose.isolated_joint}: 방향은 맞습니다. 같은 쪽으로 조금 더 움직이세요."
        return f"OK: {pose.isolated_joint} 방향과 이동량이 맞습니다."
    changed = [
        JOINT_ORDER[index]
        for index, delta in enumerate(deltas)
        if abs(delta) >= MIN_ISOLATED_DELTA_DEGREES
    ]
    if len(changed) < 2:
        return "조합 자세: 편한 방향으로 관절을 하나 더 움직이세요."
    return "OK: 조합 자세가 충분히 다릅니다."


def pose_feedback(error: PoseRejectedError) -> str:
    """Translate strict pose validation into one plain physical correction."""
    detail = str(error)
    if "wrong sign" in detail:
        return "반대 방향입니다. 기준 자세를 지나 반대쪽으로 같은 관절만 움직이세요."
    if "insufficient span" in detail:
        return "방향은 맞지만 이동이 작습니다. 같은 방향으로 조금 더 움직이세요."
    if "another joint" in detail:
        return "다른 관절도 많이 움직였습니다. 기준 자세로 돌아가 지정 관절 하나만 움직이세요."
    if "duplicate/static" in detail:
        return "거의 움직이지 않았습니다. 지정 관절을 눈에 보이게 조금 더 움직이세요."
    if "combination pose" in detail:
        return "조합 자세입니다. 편하게 두 관절 이상을 조금씩 움직이세요."
    return detail


_JOINT_KOREAN = {
    "shoulder_pan": "맨 아래 베이스 회전축(shoulder_pan)",
    "shoulder_lift": "베이스 바로 위 어깨축(shoulder_lift)",
    "elbow_flex": "팔 중간 팔꿈치축(elbow_flex)",
    "wrist_flex": "손목을 위아래로 꺾는 축(wrist_flex)",
    "wrist_roll": "손목을 비트는 축(wrist_roll)",
}


def korean_pose_instruction(pose: PoseInstruction) -> str:
    """Describe one pose in plain Korean for the physical operator."""
    if pose.identifier == "fit-baseline":
        return "편안하고 손을 떼도 유지되는 중앙 기준 자세를 만드세요."
    if pose.isolated_joint is not None:
        joint = _JOINT_KOREAN[pose.isolated_joint]
        direction = (
            "아무 한쪽으로 5~10도" if pose.direction == -1 else "기준 자세의 반대편으로 5~10도"
        )
        return f"{joint} 하나만 {direction} 움직이세요."
    descriptions = {
        "fit-task-left": "그리퍼 끝을 작업영역 왼쪽에 두되 관절 두 개 이상을 편하게 조합하세요.",
        "fit-task-right": "그리퍼 끝을 작업영역 오른쪽에 두되 앞 자세와 다르게 조합하세요.",
        "held-task-a": "앞에서 쓰지 않은 새로운 중앙 작업 자세를 만드세요.",
        "held-task-b": "이전 모든 자세와 다른 두 번째 중앙 작업 자세를 만드세요.",
    }
    return descriptions[pose.identifier]


def korean_current_pose_message(pose: PoseInstruction) -> str:
    index = next(
        index for index, planned in enumerate(POSE_PLAN, 1) if planned.identifier == pose.identifier
    )
    return f"[현재 {index}/15] {korean_pose_instruction(pose)}"


def korean_progress_message(saved_count: int) -> str:
    saved_pose = POSE_PLAN[saved_count - 1]
    saved_name = "기준 자세" if saved_pose.identifier == "fit-baseline" else saved_pose.identifier
    message = f"[저장 완료 {saved_count}/15] {saved_name}를 저장했습니다."
    if saved_count == len(POSE_PLAN):
        return message + "\n[완료] 15개 자세를 모두 저장했습니다."
    return (
        message + f"\n[다음 {saved_count + 1}/15] {korean_pose_instruction(POSE_PLAN[saved_count])}"
    )


def korean_rejection_message(saved_count: int, guidance: str) -> str:
    """Explain why nothing was saved and exactly how to retry."""
    return (
        f"[저장 안 됨 {saved_count}/15] 현재 자세가 조건에 맞지 않습니다.\n"
        f"[문제와 수정 방법] {guidance}\n"
        "[다시 시도] 자세를 고치고 손을 완전히 뗀 뒤 Enter를 누르세요."
    )


def check_violation_feedback(violation: RolloutViolation) -> str | None:
    """Translate recoverable CHECK-only range failures into operator guidance."""
    if violation.code is not RolloutCode.R_OUT_OF_RANGE:
        return None
    detail = str(violation).partition(": ")[2]
    return (
        "허용 범위를 넘었습니다. 같은 방향으로 더 움직이지 말고 반대쪽으로 "
        f"되돌린 뒤 다시 check 하세요. {detail}"
    )


def run_guided_capture(request: GuidedCaptureRequest) -> CaptureOutcome:
    """Preflight torque, then capture one explicit confirmation event per pose."""
    session_root = request.session_root
    identity = request.identity
    policy = request.policy
    bus = request.bus
    confirm = request.confirm
    clock = request.clock
    wall_clock = request.wall_clock
    follower_permissions = request.follower_permissions
    preflight_only = request.preflight_only
    if TORQUE_PERMISSION not in follower_permissions:
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, TORQUE_NOT_PROVEN_INSTRUCTION)
    if identity.provider_digest != JOINT_CORPUS_PROVIDER_DIGEST:
        raise RolloutViolation(RolloutCode.R_PROVIDER_MISMATCH, "joint corpus provider mismatch")
    session_digest = hashlib.sha256(canonical_json(_identity_document(identity))).hexdigest()
    calibration = _calibration(identity.calibration_path)
    if identity.production:
        _verify_live_identity(identity)
    opened = False
    prior: list[Mapping[str, object]] = []
    corpus_path: Path | None = None
    try:
        bus.connect()
        opened = True
        prove_manual_positioning_safe(bus, follower_permissions)
        if preflight_only:
            return CaptureOutcome(
                "PREFLIGHT_OK", 0, len(POSE_PLAN), POSE_PLAN[0].identifier, None, session_digest
            )
        if policy is None:
            raise RolloutViolation(
                RolloutCode.R_POLICY_UNAUTHORIZED,
                "capture requires a current owner-signed production safety policy",
            )
        members_root, session_digest = _prepare_session(session_root, identity)
        prior = _existing_members(session_root)
        for pose in POSE_PLAN[len(prior) :]:
            while True:
                event = confirm(pose)
                if event == "STOP":
                    raise CaptureStoppedError
                expected = f"CAPTURE {pose.identifier}"
                check_only = event in {"CHECK", f"CHECK {pose.identifier}"}
                if event != expected and not check_only:
                    raise PoseRejectedError(f"confirmation must be CHECK or exactly: {expected}")
                if identity.production:
                    _verify_live_identity(identity)
                prove_manual_positioning_safe(bus, follower_permissions)
                confirmed_at = wall_clock()
                started = clock()
                observed = bus.sync_read("Present_Position", normalize=False)
                completed = clock()
                raw_counts = _read_vector(observed, "Present_Position")
                derived_at = clock()
                try:
                    document = _member(
                        MemberCaptureInput(
                            pose,
                            raw_counts,
                            calibration,
                            policy,
                            identity,
                            event,
                            confirmed_at,
                            started,
                            completed,
                            derived_at,
                        )
                    )
                except RolloutViolation as exc:
                    feedback = check_violation_feedback(exc) if identity.production else None
                    if feedback is None:
                        raise
                    print(korean_rejection_message(len(prior), feedback))
                    continue
                physical = cast("Mapping[str, object]", document["physical"])
                degrees = cast("Sequence[float]", physical["joint_degrees"])
                try:
                    validate_pose(pose, degrees, prior)
                except PoseRejectedError as exc:
                    guidance = (
                        pose_check_message(pose, degrees, prior) if prior else pose_feedback(exc)
                    )
                    print(korean_rejection_message(len(prior), guidance))
                    continue
                if check_only:
                    print(f"[검사 통과] {pose_check_message(pose, degrees, prior)}")
                    print("[저장하려면] 손을 완전히 떼고 s를 누르세요.")
                    continue
                _publish(members_root / f"{pose.identifier}.json", document)
                prior.append(document)
                print(korean_progress_message(len(prior)))
                break
    finally:
        if opened:
            bus.disconnect(disable_torque=False)
    if len(prior) != len(POSE_PLAN):
        raise RolloutViolation(RolloutCode.EQUIVALENCE_UNPROVEN, "partial session is incomplete")
    manifest = _manifest(prior, calibration, policy, identity)
    corpus_path = session_root / "corpus.json"
    if corpus_path.exists():
        if _load_member(corpus_path) != manifest:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "completed corpus drift")
    else:
        _publish(corpus_path, manifest)
    return CaptureOutcome(
        "CAPTURE_COMPLETE_OWNER_SIGNATURE_REQUIRED",
        len(prior),
        len(POSE_PLAN),
        None,
        corpus_path,
        session_digest,
    )


def identity_from_authority(
    authority: ProductionReadOnlyAcquisitionAuthority,
    policy: ProductionApprovedSafetyPolicy,
    *,
    capture_id: str,
) -> CaptureIdentity:
    """Bind a production session to verified signed read-only authority bytes."""
    return CaptureIdentity(
        authority.canonical_digest,
        policy.canonical_digest,
        authority.provider_digest,
        authority.follower_device_digest,
        authority.calibration_digest,
        capture_id,
        authority.profile_path,
        authority.profile_digest,
        authority.calibration_path,
        authority.follower_device_path,
        True,
    )
