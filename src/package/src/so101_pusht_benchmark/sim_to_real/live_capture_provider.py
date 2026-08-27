"""Owner-integrated real factories for the governed read-only live capture route."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from importlib import import_module
from pathlib import Path
import time
from typing import cast, Protocol

from .live_capture_adapters import (
    CaptureFactory,
    DirectBusJointReader,
    DirectReadRobot,
    ReadOnlyOpenCvCamera,
)
from .live_capture_cli import LiveCaptureDependencies
from .live_capture_identity import ApprovedLiveIdentity, load_approved_live_identity
from .live_capture_process import MultiprocessingProviderRuntime
from .live_capture_runtime import verify_feetech_runtime
from .live_capture_types import AdapterIdentity, LiveCaptureConfiguration
from .policy_approval import ProductionTrustStore
from .policy_parser import load_production_safety_policy
from .policy_types import ProductionApprovedSafetyPolicy
from .read_only_authority import (
    ProductionReadOnlyAcquisitionAuthority,
    load_read_only_acquisition_authority,
    path_metadata_digest,
)
from .rollout_codes import RolloutCode, RolloutViolation
from .sample_capture import Clock

__all__ = (
    "LIVE_READ_PROVIDER_DIGEST",
    "build_governed_live_dependencies",
    "probe_device_identity",
    "sha256_file",
)
_PROVIDER_ID = b"so101-pusht-benchmark:preflighted-process-read-only-live-provider:v3"
LIVE_READ_PROVIDER_DIGEST = hashlib.sha256(_PROVIDER_ID).hexdigest()


class _Cv2Module(Protocol):
    CAP_PROP_FRAME_WIDTH: int
    CAP_PROP_FRAME_HEIGHT: int
    CAP_PROP_FPS: int

    def VideoCapture(self, path: str) -> object: ...


class _ConfigFactory(Protocol):
    def __call__(
        self,
        *,
        port: str,
        id: str,
        cameras: dict[str, object],
        use_degrees: bool,
    ) -> object: ...


class _FollowerFactory(Protocol):
    def __call__(self, config: object) -> object: ...


def sha256_file(path: Path) -> str:
    """Hash one existing regular authority file without changing it."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RolloutViolation(RolloutCode.R_MISSING, f"authority file is missing: {path}") from exc
    return digest.hexdigest()


def probe_device_identity(path: Path) -> str | None:
    """Observe current by-id metadata without opening the device."""
    try:
        return path_metadata_digest(path)
    except RolloutViolation:
        return None


def _camera_factory(clock: Clock) -> Callable[[LiveCaptureConfiguration], ReadOnlyOpenCvCamera]:
    def build(configuration: LiveCaptureConfiguration) -> ReadOnlyOpenCvCamera:
        cv2 = cast("_Cv2Module", import_module("cv2"))
        device_digest = probe_device_identity(configuration.camera_device)
        if device_digest is None:
            raise RolloutViolation(RolloutCode.R_MISSING, "camera device disappeared")
        return ReadOnlyOpenCvCamera(
            configuration,
            AdapterIdentity(LIVE_READ_PROVIDER_DIGEST, device_digest, None),
            capture_factory=cast("CaptureFactory", cv2.VideoCapture),
            property_ids=(
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            ),
            clock=clock,
        )

    return build


def _joint_factory(clock: Clock) -> Callable[[LiveCaptureConfiguration], DirectBusJointReader]:
    def build(configuration: LiveCaptureConfiguration) -> DirectBusJointReader:
        config_module = import_module("lerobot.robots.so_follower.config_so_follower")
        follower_module = import_module("lerobot.robots.so_follower.so_follower")
        config_factory = cast("_ConfigFactory", config_module.__dict__["SOFollowerRobotConfig"])
        follower_factory = cast("_FollowerFactory", follower_module.__dict__["SOFollower"])
        device_digest = probe_device_identity(configuration.follower_device)
        if device_digest is None:
            raise RolloutViolation(RolloutCode.R_MISSING, "follower device disappeared")
        calibration_digest = sha256_file(configuration.calibration_file)
        config = config_factory(
            port=str(configuration.follower_device),
            id=configuration.follower_calibration_id,
            cameras={},
            use_degrees=True,
        )
        robot = follower_factory(config)
        identity = AdapterIdentity(
            LIVE_READ_PROVIDER_DIGEST,
            device_digest,
            calibration_digest,
        )
        return DirectBusJointReader(cast("DirectReadRobot", robot), identity, clock=clock)

    return build


def build_governed_live_dependencies(
    trust_store: ProductionTrustStore,
    *,
    clock: Clock = time.monotonic,
) -> LiveCaptureDependencies:
    """Bind real adapters only to one externally governed production trust store."""
    if type(trust_store) is not ProductionTrustStore or not trust_store.is_governed():
        raise RolloutViolation(RolloutCode.R_POLICY_UNAUTHORIZED, "governed trust store required")

    def load_policy(path: Path) -> ProductionApprovedSafetyPolicy:
        return load_production_safety_policy(path, trust_store=trust_store)

    def load_identity(path: Path) -> ApprovedLiveIdentity:
        identity = load_approved_live_identity(path, trust_store=trust_store)
        if identity.provider_digest != LIVE_READ_PROVIDER_DIGEST:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "live provider identity mismatch")
        return identity

    def load_acquisition_authority(
        path: Path, signature_path: Path
    ) -> ProductionReadOnlyAcquisitionAuthority:
        authority = load_read_only_acquisition_authority(
            path, signature_path=signature_path, trust_store=trust_store
        )
        if authority.provider_digest != LIVE_READ_PROVIDER_DIGEST:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "live provider identity mismatch")
        return authority

    return LiveCaptureDependencies(
        policy_loader=load_policy,
        identity_loader=load_identity,
        camera_factory=_camera_factory(clock),
        joint_factory=_joint_factory(clock),
        device_probe=probe_device_identity,
        profile_digest=sha256_file,
        calibration_digest=sha256_file,
        clock=clock,
        process_runtime=MultiprocessingProviderRuntime(),
        runtime_preflight=verify_feetech_runtime,
        acquisition_authority_loader=load_acquisition_authority,
    )
