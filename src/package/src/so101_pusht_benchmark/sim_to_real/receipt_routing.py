"""Fail-closed lexical and resolved routing for guarded-rollout receipts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import stat
from typing import cast, Final

import yaml

__all__ = (
    "CANONICAL_ROLLOUT_ROOT",
    "ReceiptPathIdentity",
    "ReceiptRoutingError",
    "locate_receipt_path",
    "prepare_receipt_directory",
    "validate_receipt_identity",
    "validate_receipt_path",
)

_PROJECT_ROOT: Final = Path("/home/intro/InternLab/02_InTro_Project")
_ARTIFACT_ALIAS: Final = _PROJECT_ROOT / "04_experiments"
_DECLARATION: Final = (
    _PROJECT_ROOT / "03_code/so101_pusht_benchmark/configs/provenance/external_inputs.yaml"
)
CANONICAL_ROLLOUT_ROOT: Final = (
    _ARTIFACT_ALIAS / "so101_pusht_benchmark/inference/sim_to_real_rollout"
)


class ReceiptRoutingError(ValueError):
    """Raised before a receipt can be read or written through an unsafe path."""


@dataclass(frozen=True, slots=True)
class ReceiptPathIdentity:
    """Trusted lexical authority and its separately verified IO target."""

    lexical: Path
    resolved: Path
    canonical: bool


def _declared_alias_target() -> Path:
    try:
        parsed: object = yaml.safe_load(_DECLARATION.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ReceiptRoutingError("canonical alias declaration must be a mapping")
        raw = cast("dict[object, object]", parsed)
        aliases_raw = raw.get("trusted_path_aliases")
        if not isinstance(aliases_raw, dict):
            raise ReceiptRoutingError("canonical alias declaration is incomplete")
        aliases = cast("dict[object, object]", aliases_raw)
        if set(aliases) != {"04_experiments"}:
            raise ReceiptRoutingError("canonical alias declaration is incomplete")
        target = aliases["04_experiments"]
        if not isinstance(target, str):
            raise ReceiptRoutingError("canonical alias target must be text")
        declared = Path(target)
    except (OSError, yaml.YAMLError) as exc:
        raise ReceiptRoutingError("cannot read canonical alias declaration") from exc
    if not declared.is_absolute() or ".." in declared.parts:
        raise ReceiptRoutingError("canonical alias target is unsafe")
    return declared


def _absolute_lexical(path: Path) -> Path:
    expanded = path.expanduser()
    if ".." in expanded.parts:
        raise ReceiptRoutingError(f"path traversal is forbidden: {path}")
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _validate_alias() -> Path:
    declared = _declared_alias_target()
    try:
        alias_info = _ARTIFACT_ALIAS.lstat()
    except OSError as exc:
        raise ReceiptRoutingError("canonical artifact alias is unavailable") from exc
    if not stat.S_ISLNK(alias_info.st_mode):
        raise ReceiptRoutingError("canonical artifact alias is not the declared symlink")
    try:
        actual = (_ARTIFACT_ALIAS.parent / _ARTIFACT_ALIAS.readlink()).resolve(strict=True)
        expected = declared.resolve(strict=True)
    except OSError as exc:
        raise ReceiptRoutingError("canonical artifact alias target is unavailable") from exc
    if actual != expected:
        raise ReceiptRoutingError("canonical artifact alias target drifted")
    return expected


def _reject_symlink_components(path: Path, *, allowed_alias: Path | None) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            if allowed_alias is None or current != allowed_alias:
                raise ReceiptRoutingError(f"symlink path component is forbidden: {current}")
        elif current != path and not stat.S_ISDIR(info.st_mode):
            raise ReceiptRoutingError(f"non-directory receipt parent: {current}")


def locate_receipt_path(path: Path) -> ReceiptPathIdentity:
    """Bind lexical authority to verified resolved IO without conflating them."""
    lexical = _absolute_lexical(path)
    canonical = _inside(lexical, CANONICAL_ROLLOUT_ROOT)
    if canonical:
        alias_target = _validate_alias()
        _reject_symlink_components(lexical, allowed_alias=_ARTIFACT_ALIAS)
        resolved = lexical.resolve(strict=False)
        resolved_root = (
            alias_target / "so101_pusht_benchmark/inference/sim_to_real_rollout"
        ).resolve(strict=False)
        if not _inside(resolved, resolved_root):
            raise ReceiptRoutingError("production receipt escapes declared alias target")
    else:
        _reject_symlink_components(lexical, allowed_alias=None)
        resolved = lexical.resolve(strict=False)
        canonical_target = CANONICAL_ROLLOUT_ROOT.resolve(strict=False)
        if _inside(resolved, canonical_target):
            raise ReceiptRoutingError("resolved path enters canonical rollout target without alias")
    try:
        info = lexical.lstat()
    except FileNotFoundError:
        return ReceiptPathIdentity(lexical, resolved, canonical)
    if stat.S_ISLNK(info.st_mode):
        raise ReceiptRoutingError(f"receipt destination cannot be a symlink: {lexical}")
    return ReceiptPathIdentity(lexical, resolved, canonical)


def validate_receipt_identity(
    identity: ReceiptPathIdentity,
    *,
    production: bool,
) -> ReceiptPathIdentity:
    """Require derived evidence scope to agree with the trusted path location."""
    current = locate_receipt_path(identity.lexical)
    if current != identity:
        raise ReceiptRoutingError("receipt path identity changed after validation")
    if production and not current.canonical:
        raise ReceiptRoutingError(
            f"production receipt must be under canonical rollout root: {CANONICAL_ROLLOUT_ROOT}"
        )
    if not production and current.canonical:
        raise ReceiptRoutingError("fixture-only receipt cannot enter the canonical rollout root")
    return current


def validate_receipt_path(path: Path, *, production: bool) -> Path:
    """Return lexical authority only after both lexical and resolved checks pass."""
    identity = locate_receipt_path(path)
    return validate_receipt_identity(identity, production=production).lexical


def prepare_receipt_directory(path: Path, *, production: bool) -> Path:
    """Create through verified IO target while retaining lexical path authority."""
    identity = validate_receipt_identity(locate_receipt_path(path), production=production)
    identity.resolved.mkdir(parents=True, exist_ok=True)
    validated = validate_receipt_identity(locate_receipt_path(path), production=production)
    if not validated.resolved.is_dir():
        raise ReceiptRoutingError(f"receipt destination is not a directory: {validated.lexical}")
    return validated.lexical
