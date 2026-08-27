"""CLI for guided, manually positioned, read-only joint/FK corpus capture."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import import_module
import json
from pathlib import Path
import sys
import time
from typing import cast, Protocol

from so101_pusht_benchmark.hardware_profile import load_hardware_profile

from .joint_corpus_capture import (
    CaptureIdentity,
    CaptureOutcome,
    CapturePolicy,
    CaptureStoppedError,
    GuidedCaptureRequest,
    RawReadBus,
    korean_current_pose_message,
    run_guided_capture,
)
from .joint_corpus_contract import (
    JOINT_CORPUS_PROVIDER_DIGEST,
    POSE_PLAN,
    TORQUE_NOT_PROVEN_INSTRUCTION,
    TORQUE_PERMISSION,
    PoseInstruction,
    render_operator_checklist,
)
from .joint_positioning_authority import load_joint_positioning_authority
from .policy_approval import ProductionTrustStore, RsaPkcs1v15Sha256Anchor
from .policy_parser import load_fixture_safety_policy, load_production_safety_policy
from .read_only_authority import load_read_only_acquisition_authority
from .receipt_routing import prepare_receipt_directory


class _ConfigFactory(Protocol):
    def __call__(
        self, *, port: str, id: str, cameras: dict[str, object], use_degrees: bool
    ) -> object: ...


class _FollowerFactory(Protocol):
    def __call__(self, config: object) -> object: ...


class _Robot(Protocol):
    bus: RawReadBus


def normalize_confirmation(value: str, pose_identifier: str | None = None) -> str:
    """Accept full commands plus one-key, one-handed operator shortcuts."""
    stripped = value.strip()
    if stripped == "" and pose_identifier is not None:
        return f"CAPTURE {pose_identifier}"
    if stripped in {"c", "C"}:
        return "CHECK"
    if stripped in {"q", "Q"}:
        return "STOP"
    if stripped in {"s", "S"} and pose_identifier is not None:
        return f"CAPTURE {pose_identifier}"
    verb, separator, remainder = stripped.partition(" ")
    if verb.casefold() in {"check", "capture", "stop"}:
        normalized = verb.upper()
        return normalized + (separator + remainder if separator else "")
    return stripped


class KeyboardConfirmation:
    def __call__(self, pose: PoseInstruction) -> str:
        print(f"\n{korean_current_pose_message(pose)}")
        print("자세를 만든 뒤 손을 완전히 떼세요.")
        print("Enter 또는 s = 검사 후 맞으면 즉시 저장. c = 검사만. q = 중단.")
        return normalize_confirmation(
            input("[Enter/s] 검사+저장 | [c] 검사만 | [q] 중단: "), pose.identifier
        )


class _FixtureConfirmation:
    def __init__(self, stop_after: int | None) -> None:
        self._count = 0
        self._stop_after = stop_after

    def __call__(self, pose: PoseInstruction) -> str:
        if self._stop_after is not None and self._count >= self._stop_after:
            return "STOP"
        self._count += 1
        return f"CAPTURE {pose.identifier}"


class _FixtureBus:
    def __init__(self, document: Mapping[str, object], *, skip_samples: int = 0) -> None:
        samples = document.get("raw_encoder_samples")
        torque = document.get("torque_enabled")
        if not isinstance(samples, list) or not isinstance(torque, list):
            raise TypeError("fixture requires raw_encoder_samples and torque_enabled")
        self._samples = iter(cast("list[list[int]]", samples)[skip_samples:])
        self._torque = cast("list[int]", torque)
        self._open = False

    def connect(self) -> None:
        self._open = True

    def sync_read(
        self,
        register: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
    ) -> Mapping[str, int | float]:
        del motors
        if not self._open or normalize:
            raise RuntimeError("fixture bus requires one open raw read")
        motor_names = (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )
        if register == "Torque_Enable":
            if len(self._torque) != 6:
                raise RuntimeError("fixture torque vector must contain six values")
            return dict(zip(motor_names, self._torque, strict=True))
        if register != "Present_Position":
            raise RuntimeError("fixture register is forbidden")
        try:
            values = next(self._samples)
        except StopIteration as exc:
            raise RuntimeError("fixture encoder stream exhausted after rejected pose") from exc
        if len(values) != 6:
            raise RuntimeError("fixture encoder vector must contain six values")
        return dict(zip(motor_names, values, strict=True))

    def disconnect(self, *, disable_torque: bool) -> None:
        if disable_torque:
            raise AssertionError("torque write requested")
        self._open = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Guide 15 manually positioned poses and publish only an unsigned, resumable "
            "joint/FK corpus candidate. No automatic movement or hardware writes."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="use the governed direct follower bus")
    mode.add_argument("--fixture", type=Path, help="use an injected fake read stream")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--acquisition-authority", type=Path)
    parser.add_argument("--authority-signature", type=Path)
    parser.add_argument("--positioning-authority", type=Path)
    parser.add_argument("--positioning-signature", type=Path)
    parser.add_argument("--trust-anchor", type=Path)
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--capture-id")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify authority/identity and read Torque_Enable; capture no pose",
    )
    parser.add_argument(
        "--checklist-output",
        type=Path,
        help="write the generated operator checklist and exit without device access",
    )
    parser.add_argument(
        "--fixture-stop-after",
        type=int,
        help="fixture-only explicit STOP after N newly confirmed poses",
    )
    return parser


def _required_path(value: object, flag: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{flag} is required")
    return value


def _required_text(value: object, flag: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{flag} is required")
    return value


def _trust(path: Path) -> ProductionTrustStore:
    anchor = RsaPkcs1v15Sha256Anchor.from_pem_file(path)
    return ProductionTrustStore.from_owner_anchors((anchor,))


def _live_bus(profile_path: Path) -> RawReadBus:
    profile = load_hardware_profile(profile_path)
    config_module = import_module("lerobot.robots.so_follower.config_so_follower")
    follower_module = import_module("lerobot.robots.so_follower.so_follower")
    config_factory = cast("_ConfigFactory", config_module.__dict__["SOFollowerRobotConfig"])
    follower_factory = cast("_FollowerFactory", follower_module.__dict__["SOFollower"])
    config = config_factory(
        port=str(profile.follower.port),
        id=profile.follower.calibration_id,
        cameras={},
        use_degrees=True,
    )
    return cast("_Robot", follower_factory(config)).bus


def _fixture_inputs(
    fixture_path: Path, policy_path: Path, session: Path, capture_id: str
) -> tuple[CaptureIdentity, CapturePolicy, _FixtureBus]:
    raw: object = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("fixture stream must be a mapping")
    document = cast("Mapping[str, object]", raw)
    calibration_raw = document.get("calibration")
    if not isinstance(calibration_raw, Mapping):
        raise TypeError("fixture calibration must be a mapping")
    prepare_receipt_directory(session, production=False)
    calibration = session / "fixture-calibration.json"
    encoded = (json.dumps(calibration_raw, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if calibration.exists() and calibration.read_bytes() != encoded:
        raise ValueError("fixture calibration drift")
    if not calibration.exists():
        calibration.write_bytes(encoded)
    policy = load_fixture_safety_policy(policy_path)
    import hashlib

    identity = CaptureIdentity(
        "f" * 64,
        policy.canonical_digest,
        JOINT_CORPUS_PROVIDER_DIGEST,
        hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        hashlib.sha256(calibration.read_bytes()).hexdigest(),
        capture_id,
        policy_path,
        hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        calibration,
        fixture_path,
        False,
    )
    captured = sum(
        (session / "members" / f"{pose.identifier}.json").is_file() for pose in POSE_PLAN
    )
    return identity, policy, _FixtureBus(document, skip_samples=captured)


def _outcome_document(outcome: CaptureOutcome) -> dict[str, object]:
    return {
        "status": outcome.status,
        "captured_count": outcome.captured_count,
        "required_count": outcome.required_count,
        "next_pose": outcome.next_pose,
        "corpus_path": str(outcome.corpus_path) if outcome.corpus_path is not None else None,
        "session_identity_digest": outcome.session_identity_digest,
        "genuine_scope_granted": False,
        "owner_signature_required": outcome.corpus_path is not None,
        "motor_writes_performed": False,
        "actuation_performed": False,
    }


def run_joint_corpus_capture_cli(argv: list[str] | None = None) -> int:
    """Route checklist, injected fixture, or owner-governed live capture."""
    args = _parser().parse_args(argv)
    try:
        if isinstance(args.checklist_output, Path):
            args.checklist_output.write_text(render_operator_checklist(), encoding="utf-8")
            print(args.checklist_output)
            return 0
        policy: CapturePolicy | None
        if isinstance(args.fixture, Path):
            session = _required_path(args.session_dir, "--session-dir")
            policy_path = _required_path(args.policy, "--policy")
            capture_id = _required_text(args.capture_id, "--capture-id")
            identity, policy, bus = _fixture_inputs(args.fixture, policy_path, session, capture_id)
            confirmation = _FixtureConfirmation(args.fixture_stop_after)
            permissions = ("direct_bus_connect", TORQUE_PERMISSION, "sync_read:Present_Position")
        elif args.live:
            profile = _required_path(args.profile, "--profile")
            authority_path = _required_path(args.acquisition_authority, "--acquisition-authority")
            signature = _required_path(args.authority_signature, "--authority-signature")
            trust_path = _required_path(args.trust_anchor, "--trust-anchor")
            trust = _trust(trust_path)
            authority = load_read_only_acquisition_authority(
                authority_path, signature_path=signature, trust_store=trust
            )
            if not isinstance(args.positioning_authority, Path) or not isinstance(
                args.positioning_signature, Path
            ):
                raise TypeError(TORQUE_NOT_PROVEN_INSTRUCTION)
            positioning = load_joint_positioning_authority(
                args.positioning_authority,
                signature_path=args.positioning_signature,
                trust_store=trust,
                base=authority,
            )
            if args.preflight_only:
                session = Path("/preflight-only-session-not-written")
                policy = None
                capture_id = "preflight-only"
            else:
                session = _required_path(args.session_dir, "--session-dir")
                policy_path = _required_path(args.policy, "--policy")
                capture_id = _required_text(args.capture_id, "--capture-id")
                policy = load_production_safety_policy(policy_path, trust_store=trust)
            identity = CaptureIdentity(
                positioning.canonical_digest,
                policy.canonical_digest if policy is not None else "0" * 64,
                positioning.provider_digest,
                positioning.follower_device_digest,
                positioning.calibration_digest,
                capture_id,
                authority.profile_path,
                authority.profile_digest,
                authority.calibration_path,
                authority.follower_device_path,
                True,
            )
            if not args.preflight_only:
                assert policy is not None
            if profile.resolve(strict=True) != authority.profile_path:
                raise ValueError("--profile does not match the signed authority profile")
            bus = _live_bus(profile)
            confirmation = KeyboardConfirmation()
            permissions = authority.follower_permissions
        else:
            raise ValueError("exactly one of --live or --fixture is required")
        outcome = run_guided_capture(
            GuidedCaptureRequest(
                session,
                identity,
                policy,
                bus,
                confirmation,
                time.monotonic,
                lambda: datetime.now(timezone.utc),
                permissions,
                bool(args.preflight_only),
            )
        )
    except CaptureStoppedError:
        print(
            "PARTIAL: session preserved for resume; it is not complete or acceptable evidence.",
            file=sys.stderr,
        )
        return 3
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(_outcome_document(outcome), indent=2, sort_keys=True))
    return 0


def main() -> int:
    return run_joint_corpus_capture_cli()
