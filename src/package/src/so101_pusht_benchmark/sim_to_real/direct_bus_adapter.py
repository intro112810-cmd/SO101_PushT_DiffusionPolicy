"""Real SO-101 construction adapter for the sole direct-bus writer capability."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast

from so101_pusht_benchmark.hardware_profile import HardwareProfile

from .arming import ArmingResult
from .rollout_codes import RolloutCode, RolloutViolation
from .single_step_authorization import SingleStepAuthorization
from .writer import DirectBusRobot

__all__ = ("RealRobotFactory", "build_real_direct_bus_adapter", "lerobot_so101_factory")


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


class RealRobotFactory(Protocol):
    """Construct an already-configured robot object without opening its bus."""

    def __call__(self, profile: HardwareProfile) -> DirectBusRobot:
        """Return only the robot wrapper consumed by DirectBusWriter."""
        ...


def build_real_direct_bus_adapter(
    profile: HardwareProfile,
    authorization: SingleStepAuthorization,
    armed: ArmingResult,
    factory: RealRobotFactory,
) -> DirectBusRobot:
    """Construct the real adapter only after exact production evidence binding."""
    if authorization.artifact_scope != "production":
        raise RolloutViolation(
            RolloutCode.R_POLICY_UNAUTHORIZED, "production authorization required"
        )
    if (
        not armed.armed
        or armed.motor_writes_performed
        or authorization.digest != armed.authorization_digest
        or authorization.armed_receipt_digest != armed.receipt_digest
        or authorization.policy_digest != profile.policy_digest
        or authorization.proposal_hash != armed.proposal_hash
        or authorization.command_id != armed.command_id
    ):
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "real adapter evidence binding")
    return factory(profile)


def lerobot_so101_factory(profile: HardwareProfile) -> DirectBusRobot:
    """Construct LeRobot's calibrated wrapper; the sole writer opens only ``robot.bus``."""
    config_module = import_module("lerobot.robots.so_follower.config_so_follower")
    follower_module = import_module("lerobot.robots.so_follower.so_follower")
    config_factory = cast("_ConfigFactory", config_module.__dict__["SOFollowerRobotConfig"])
    follower_factory = cast("_FollowerFactory", follower_module.__dict__["SOFollower"])
    follower = profile.follower
    config = config_factory(
        port=str(follower.port),
        id=follower.calibration_id,
        cameras={},
        use_degrees=True,
    )
    return cast("DirectBusRobot", follower_factory(config))
