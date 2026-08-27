"""Signed runtime identity for the process-isolated follower read provider."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata, util
from pathlib import Path
from typing import Final

from .rollout_codes import RolloutCode, RolloutViolation

__all__ = (
    "RUNTIME_FIELDS",
    "ObservedAuthorityRuntime",
    "observe_authority_runtime",
)
_FEETECH_DISTRIBUTION: Final = "feetech-servo-sdk"
_FEETECH_VERSION: Final = "1.0.0"
_PYSERIAL_DISTRIBUTION: Final = "pyserial"
_PYSERIAL_VERSION: Final = "3.5"
_SCSERVO_MODULE: Final = "scservo_sdk"
RUNTIME_FIELDS = frozenset(
    {
        "feetech_servo_sdk_distribution",
        "feetech_servo_sdk_version",
        "pyserial_distribution",
        "pyserial_version",
        "scservo_sdk_distribution",
        "scservo_sdk_module",
        "scservo_sdk_origin",
        "scservo_sdk_origin_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class ObservedAuthorityRuntime:
    """Current import/distribution identity derived without opening a device."""

    feetech_servo_sdk_distribution: str
    feetech_servo_sdk_version: str
    pyserial_distribution: str
    pyserial_version: str
    scservo_sdk_distribution: str
    scservo_sdk_module: str
    scservo_sdk_origin: Path
    scservo_sdk_origin_sha256: str

    def as_document(self) -> dict[str, object]:
        """Return canonical JSON-compatible signed runtime fields."""
        return {
            "feetech_servo_sdk_distribution": self.feetech_servo_sdk_distribution,
            "feetech_servo_sdk_version": self.feetech_servo_sdk_version,
            "pyserial_distribution": self.pyserial_distribution,
            "pyserial_version": self.pyserial_version,
            "scservo_sdk_distribution": self.scservo_sdk_distribution,
            "scservo_sdk_module": self.scservo_sdk_module,
            "scservo_sdk_origin": str(self.scservo_sdk_origin),
            "scservo_sdk_origin_sha256": self.scservo_sdk_origin_sha256,
        }


def _version(distribution: str, expected: str) -> str:
    try:
        observed = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise RolloutViolation(
            RolloutCode.R_PROVIDER_MISMATCH,
            f"required runtime distribution is missing: {distribution}",
        ) from exc
    if observed != expected:
        raise RolloutViolation(
            RolloutCode.R_PROVIDER_MISMATCH,
            f"runtime distribution drift: {distribution}=={observed}",
        )
    return observed


def _module_origin() -> Path:
    owners = tuple(metadata.packages_distributions().get(_SCSERVO_MODULE, ()))
    if owners != (_FEETECH_DISTRIBUTION,):
        raise RolloutViolation(
            RolloutCode.R_PROVIDER_MISMATCH,
            "scservo_sdk distribution ownership drift",
        )
    specification = util.find_spec(_SCSERVO_MODULE)
    raw = None if specification is None else specification.origin
    if raw is None:
        raise RolloutViolation(RolloutCode.R_PROVIDER_MISMATCH, "scservo_sdk origin is missing")
    origin = Path(raw)
    if not origin.is_absolute() or not origin.is_file() or origin.is_symlink():
        raise RolloutViolation(RolloutCode.R_PROVIDER_MISMATCH, "scservo_sdk origin is invalid")
    return origin


def observe_authority_runtime() -> ObservedAuthorityRuntime:
    """Verify and fingerprint the exact SDK/serial runtime before signing or capture."""
    feetech_version = _version(_FEETECH_DISTRIBUTION, _FEETECH_VERSION)
    pyserial_version = _version(_PYSERIAL_DISTRIBUTION, _PYSERIAL_VERSION)
    origin = _module_origin()
    return ObservedAuthorityRuntime(
        _FEETECH_DISTRIBUTION,
        feetech_version,
        _PYSERIAL_DISTRIBUTION,
        pyserial_version,
        _FEETECH_DISTRIBUTION,
        _SCSERVO_MODULE,
        origin,
        hashlib.sha256(origin.read_bytes()).hexdigest(),
    )
