"""Pre-device Feetech runtime dependency identity verification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol

from .rollout_codes import RolloutCode, RolloutViolation

__all__ = (
    "FEETECH_DISTRIBUTION",
    "FEETECH_MODULE",
    "FEETECH_VERSION",
    "RuntimeDependencyReceipt",
    "RuntimeInspector",
    "RuntimePreflight",
    "verify_feetech_runtime",
)
FEETECH_DISTRIBUTION: Final = "feetech-servo-sdk"
FEETECH_MODULE: Final = "scservo_sdk"
FEETECH_VERSION: Final = "1.0.0"


@dataclass(frozen=True, slots=True)
class RuntimeDependencyReceipt:
    distribution: str
    version: str
    module: str
    module_origin: Path


class RuntimePreflight(Protocol):
    def __call__(self) -> RuntimeDependencyReceipt: ...


def _installed_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _module_distributions(name: str) -> tuple[str, ...]:
    return tuple(packages_distributions().get(name, ()))


def _load_module(name: str) -> ModuleType | None:
    try:
        return import_module(name)
    except ImportError:
        return None


@dataclass(frozen=True, slots=True)
class RuntimeInspector:
    installed_version: Callable[[str], str | None]
    module_distributions: Callable[[str], tuple[str, ...]]
    load_module: Callable[[str], ModuleType | None]


def verify_feetech_runtime(
    inspector: RuntimeInspector | None = None,
) -> RuntimeDependencyReceipt:
    """Import and bind the exact SDK distribution before any provider spawn."""
    selected = (
        RuntimeInspector(_installed_version, _module_distributions, _load_module)
        if inspector is None
        else inspector
    )
    observed_version = selected.installed_version(FEETECH_DISTRIBUTION)
    if observed_version != FEETECH_VERSION:
        detail = "missing" if observed_version is None else observed_version
        raise RolloutViolation(
            RolloutCode.R_PROVIDER_MISMATCH,
            f"{FEETECH_DISTRIBUTION}=={FEETECH_VERSION} required; observed {detail}",
        )
    owners = selected.module_distributions(FEETECH_MODULE)
    if owners != (FEETECH_DISTRIBUTION,):
        raise RolloutViolation(
            RolloutCode.R_PROVIDER_MISMATCH,
            f"{FEETECH_MODULE} distribution ownership mismatch",
        )
    module = selected.load_module(FEETECH_MODULE)
    origin_raw = None if module is None else module.__file__
    if origin_raw is None:
        raise RolloutViolation(
            RolloutCode.R_PROVIDER_MISMATCH,
            f"required module is unavailable: {FEETECH_MODULE}",
        )
    origin = Path(origin_raw)
    if not origin.is_absolute() or not origin.is_file() or origin.is_symlink():
        raise RolloutViolation(
            RolloutCode.R_PROVIDER_MISMATCH,
            f"required module origin is invalid: {FEETECH_MODULE}",
        )
    return RuntimeDependencyReceipt(
        FEETECH_DISTRIBUTION,
        observed_version,
        FEETECH_MODULE,
        origin,
    )
