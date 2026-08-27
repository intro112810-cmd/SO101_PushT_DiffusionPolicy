"""CLI routing for fixture-only and governed live physical sample capture."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Protocol

from so101_pusht_benchmark.hardware_profile import load_hardware_profile

from .fixture_sample_capture import capture_fixture_receipt
from .live_capture import capture_live_samples, live_receipt
from .live_capture_failure import (
    LiveCaptureAttemptError,
    terminal_failure_receipt,
)
from .live_capture_protocol import ProviderProcessRuntime
from .live_capture_runtime import RuntimePreflight
from .live_capture_validation import require_live_identity
from .read_only_authority import require_read_only_acquisition_authority
from .live_capture_identity import ApprovedLiveIdentity
from .live_capture_types import (
    DeviceIdentityProbe,
    DigestFile,
    LiveCameraFactory,
    LiveCaptureConfiguration,
    LiveCaptureProviders,
    LiveCaptureRequest,
    LiveJointFactory,
)
from .policy_types import ProductionApprovedSafetyPolicy
from .read_only_authority import ProductionReadOnlyAcquisitionAuthority
from .receipt_routing import (
    ReceiptRoutingError,
    locate_receipt_path,
    prepare_receipt_directory,
    validate_receipt_identity,
    validate_receipt_path,
)
from .rollout_codes import RolloutCode, RolloutViolation
from .sample_capture import Clock
from .secure_io import atomic_write_new, unlink_owned_leaf

__all__ = ("LiveCaptureDependencies", "main", "publish_capture_receipt", "run_capture_cli")


class PolicyLoader(Protocol):
    def __call__(
        self, path: Path, /
    ) -> ProductionApprovedSafetyPolicy | ProductionReadOnlyAcquisitionAuthority: ...


class AcquisitionAuthorityLoader(Protocol):
    def __call__(
        self, path: Path, signature_path: Path, /
    ) -> ProductionReadOnlyAcquisitionAuthority: ...


class IdentityLoader(Protocol):
    def __call__(self, path: Path, /) -> ApprovedLiveIdentity: ...


class ReceiptPublisher(Protocol):
    def __call__(
        self,
        path: Path,
        receipt: dict[str, object],
        production: bool,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class LiveCaptureDependencies:
    """Owner-governed loaders, process runtime, and read-only adapter factories."""

    policy_loader: PolicyLoader
    identity_loader: IdentityLoader
    camera_factory: LiveCameraFactory
    joint_factory: LiveJointFactory
    device_probe: DeviceIdentityProbe
    profile_digest: DigestFile
    calibration_digest: DigestFile
    clock: Clock
    process_runtime: ProviderProcessRuntime
    runtime_preflight: RuntimePreflight
    acquisition_authority_loader: AcquisitionAuthorityLoader | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture two synchronized read-only physical samples."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", type=Path)
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--identity-evidence", type=Path)
    parser.add_argument("--acquisition-authority", type=Path)
    parser.add_argument("--authority-signature", type=Path)
    parser.add_argument("--trust-anchor", type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _live_profile(args: argparse.Namespace) -> Path:
    profile = args.profile
    if not isinstance(profile, Path):
        raise RolloutViolation(RolloutCode.R_MISSING, "--live requires --profile")
    if args.count != 2:
        raise RolloutViolation(RolloutCode.R_MISSING, "live capture requires exactly two samples")
    return profile


def _configuration(profile_path: Path) -> LiveCaptureConfiguration:
    profile = load_hardware_profile(profile_path)
    return LiveCaptureConfiguration(
        profile_path,
        profile.camera.device,
        profile.follower.port,
        profile.follower.calibration_file,
        profile.camera.width,
        profile.camera.height,
        float(profile.camera.fps),
        profile.follower.calibration_id,
    )


@dataclass(frozen=True, slots=True)
class _TerminalCaptureFailureError(RuntimeError):
    receipt: dict[str, object]
    detail: str

    def __str__(self) -> str:
        """Return the primary error detail retained in the failure receipt."""
        return self.detail


def _capture_live(
    args: argparse.Namespace,
    dependencies: LiveCaptureDependencies | None,
) -> dict[str, object]:
    profile_path = _live_profile(args)
    validate_receipt_path(args.output, production=True)
    if dependencies is None:
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED,
            "owner-governed live provider is unavailable",
        )
    authority_path = args.acquisition_authority
    signature_path = args.authority_signature
    if isinstance(authority_path, Path) or isinstance(signature_path, Path):
        if (
            not isinstance(authority_path, Path)
            or not isinstance(signature_path, Path)
            or dependencies.acquisition_authority_loader is None
            or args.policy is not None
            or args.identity_evidence is not None
        ):
            raise RolloutViolation(
                RolloutCode.R_POLICY_UNAUTHORIZED,
                "read-only acquisition requires one authority and detached signature",
            )
        policy = dependencies.acquisition_authority_loader(authority_path, signature_path)
        identity: object = policy
    else:
        if not isinstance(args.policy, Path) or not isinstance(args.identity_evidence, Path):
            raise RolloutViolation(
                RolloutCode.R_MISSING, "legacy live capture requires policy and identity evidence"
            )
        policy = dependencies.policy_loader(args.policy)
        identity = dependencies.identity_loader(args.identity_evidence)
    authority = require_read_only_acquisition_authority(policy)
    verified_identity = require_live_identity(identity)
    try:
        result = capture_live_samples(
            LiveCaptureRequest(authority, verified_identity, _configuration(profile_path)),
            LiveCaptureProviders(
                dependencies.camera_factory,
                dependencies.joint_factory,
                dependencies.device_probe,
                dependencies.profile_digest,
                dependencies.calibration_digest,
                dependencies.clock,
                dependencies.runtime_preflight,
            ),
            process_runtime=dependencies.process_runtime,
        )
    except LiveCaptureAttemptError as exc:
        receipt = terminal_failure_receipt(
            exc.failure,
            policy_digest=authority.canonical_digest,
            identity_digest=verified_identity.identity_digest,
        )
        raise _TerminalCaptureFailureError(
            receipt,
            exc.failure.primary_error.detail,
        ) from exc
    return live_receipt(result, authority, verified_identity)


def publish_capture_receipt(
    path: Path,
    receipt: dict[str, object],
    production: bool,
) -> None:
    """Atomically publish through a revalidated lexical/resolved path identity."""
    location = validate_receipt_identity(locate_receipt_path(path), production=production)
    prepare_receipt_directory(location.lexical.parent, production=production)
    location = validate_receipt_identity(locate_receipt_path(path), production=production)
    content = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    published = atomic_write_new(
        location.resolved.parent,
        location.resolved.name,
        content,
        temporary=f".{location.resolved.name}.capture-{os.getpid()}.tmp",
    )
    try:
        validate_receipt_identity(locate_receipt_path(path), production=production)
    except (ReceiptRoutingError, RolloutViolation):
        unlink_owned_leaf(published)
        raise


def _command_live_dependencies(args: argparse.Namespace) -> LiveCaptureDependencies | None:
    trust_anchor = args.trust_anchor
    if trust_anchor is None:
        return None
    if not isinstance(trust_anchor, Path):
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "invalid trust anchor path")
    from .live_capture_provider import build_governed_live_dependencies
    from .policy_approval import ProductionTrustStore, RsaPkcs1v15Sha256Anchor

    anchor = RsaPkcs1v15Sha256Anchor.from_pem_file(trust_anchor)
    return build_governed_live_dependencies(ProductionTrustStore.from_owner_anchors((anchor,)))


def run_capture_cli(
    argv: list[str] | None = None,
    *,
    live_dependencies: LiveCaptureDependencies | None = None,
    publisher: ReceiptPublisher = publish_capture_receipt,
) -> int:
    """Publish 2/2 samples or a separate terminal non-consumable failure."""
    args = _parser().parse_args(argv)
    try:
        if args.live:
            dependencies = (
                live_dependencies
                if live_dependencies is not None
                else _command_live_dependencies(args)
            )
            receipt = _capture_live(args, dependencies)
            production = True
        else:
            validate_receipt_path(args.output, production=False)
            if not isinstance(args.policy, Path):
                raise RolloutViolation(RolloutCode.R_MISSING, "fixture capture requires --policy")
            receipt = capture_fixture_receipt(args.fixture, args.policy, args.count)
            production = False
        publisher(args.output, receipt, production)
    except _TerminalCaptureFailureError as exc:
        failure_path = args.output.with_name(f"{args.output.stem}.terminal-failure.json")
        publisher(failure_path, exc.receipt, True)
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def main() -> int:
    """Default CLI has no production trust provider and therefore fails closed live."""
    return run_capture_cli()
