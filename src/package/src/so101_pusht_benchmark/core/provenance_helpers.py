"""Typed path, manifest, archive, and config-shape helpers for provenance."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import cast


class ProvenanceError(RuntimeError):
    """Raised when a pinned input cannot be trusted."""


def is_inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


TrustedPathAliases = tuple[tuple[Path, Path], ...]


def path(
    value: str,
    root: Path,
    *,
    trusted_aliases: TrustedPathAliases = (),
) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ProvenanceError(f"unsafe path: {value}")
    lexical = root / relative
    matched_alias: tuple[Path, Path] | None = None
    current = root
    for part in relative.parts:
        current /= part
        if not current.is_symlink():
            continue
        match = next((alias for alias in trusted_aliases if current == alias[0]), None)
        if match is None or current.resolve() != match[1]:
            raise ProvenanceError(f"unsafe path: {value}")
        matched_alias = match
    candidate = lexical.resolve()
    if is_inside(candidate, root):
        return candidate
    if matched_alias is not None and is_inside(candidate, matched_alias[1]):
        return candidate
    raise ProvenanceError(f"unsafe path: {value}")


def source_path(
    value: str,
    root: Path,
    *,
    trusted_aliases: TrustedPathAliases = (),
) -> Path:
    lexical = root / value
    if lexical.is_symlink():
        raise ProvenanceError(f"source root is a symlink: {lexical}")
    return path(value, root, trusted_aliases=trusted_aliases)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(root: Path) -> str:
    rows: list[str] = []
    for item in root.rglob("*"):
        if ".git" in item.relative_to(root).parts:
            continue
        if item.is_symlink():
            target = item.resolve(strict=True)
            if not is_inside(target, root):
                raise ProvenanceError(f"unsafe symlink: {item}")
        if item.is_file():
            rows.append(f"{sha256(item)}  {item.relative_to(root)}")
        elif not item.is_dir():
            raise ProvenanceError(f"unsupported tree entry: {item}")
    return "\n".join(sorted(rows, key=lambda row: row.split("  ", 1)[1])) + "\n"


def validate_archive_members(members: list[dict[str, str]], prefix: str) -> None:
    for member in members:
        name = member["name"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ProvenanceError("unsafe archive member path")
        if not (name == prefix.rstrip("/") or name.startswith(prefix)):
            raise ProvenanceError("unsafe archive member root")
        if member["type"] not in {"file", "dir"}:
            raise ProvenanceError("unsafe archive special or link member")
        target = member.get("linkname", "")
        if target and (Path(target).is_absolute() or ".." in Path(target).parts):
            raise ProvenanceError("unsafe archive link target")


def strings(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ProvenanceError(f"invalid string list: {label}")
    result: list[str] = []
    for item in cast("list[object]", value):
        if not isinstance(item, str):
            raise ProvenanceError(f"invalid string list: {label}")
        result.append(item)
    return result


def mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProvenanceError(f"invalid mapping: {label}")
    return cast("dict[str, object]", value)
