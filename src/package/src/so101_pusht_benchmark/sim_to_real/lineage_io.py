"""No-follow lineage member access and strict JSON primitives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import TypeGuard

from .lineage_types import LineageError, LineageMember, LineageRoots, Scope


def duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys at every nesting level."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LineageError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path, label: str) -> dict[str, object]:
    """Read one strict JSON object without accepting duplicate keys."""
    try:
        parsed: object = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LineageError(f"{label} is not valid JSON") from exc
    return object_mapping(parsed, label)


def _is_object_mapping(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def object_mapping(value: object, label: str) -> dict[str, object]:
    """Copy a runtime mapping into a string-keyed object mapping."""
    if not _is_object_mapping(value):
        raise LineageError(f"{label} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise LineageError(f"{label} keys must be strings")
        result[key] = item
    return result


def object_list(value: object, label: str) -> list[object]:
    """Copy a runtime list into an object list."""
    if not _is_object_list(value):
        raise LineageError(f"{label} must be a list")
    return list(value)


def digest(value: object, label: str) -> str:
    """Require one canonical lowercase SHA-256 digest."""
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LineageError(f"{label} must be a lowercase SHA-256 digest")
    return value


def relative_path(value: object, label: str) -> str:
    """Require a portable relative path without traversal."""
    if type(value) is not str or not value or "\\" in value:
        raise LineageError(f"{label} path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise LineageError(f"{label} path escapes its authority root")
    return value


def check_root(root: Path, label: str) -> Path:
    """Require a real, non-symlink authority root."""
    absolute = root.absolute()
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise LineageError(f"{label} root is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise LineageError(f"{label} root must be a real directory, not a symlink")
    return absolute


def scoped_roots(artifact_root: Path, roots: LineageRoots) -> dict[Scope, Path]:
    """Validate and return all four scope roots."""
    return {
        "artifact": check_root(artifact_root, "artifact"),
        "package": check_root(roots.package, "package"),
        "project": check_root(roots.project, "project"),
        "runtime": check_root(roots.runtime, "runtime"),
    }


def safe_file(root: Path, relative: str, label: str) -> Path:
    """Resolve a regular member while rejecting every symlink component."""
    path = root
    info = root.lstat()
    for part in PurePosixPath(relative).parts:
        path /= part
        try:
            info = path.lstat()
        except OSError as exc:
            raise LineageError(f"lineage member is missing: {label}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise LineageError(f"lineage member path contains a symlink: {label}")
    if not stat.S_ISREG(info.st_mode):
        raise LineageError(f"lineage member is not a regular file: {label}")
    return path


def safe_absolute_file(path: Path, label: str) -> Path:
    """Require an absolute regular file with no symlink in its full path."""
    absolute = path.absolute()
    relative = PurePosixPath(*absolute.parts[1:]).as_posix()
    return safe_file(Path(absolute.anchor), relative, label)


def resolve_members(
    members: tuple[LineageMember, ...], roots: dict[Scope, Path], manifest_path: Path
) -> dict[str, Path]:
    """Resolve every member before output deletion or byte validation."""
    result: dict[str, Path] = {}
    for member in members:
        path = safe_file(roots[member.scope], member.path, member.label)
        if path.absolute() == manifest_path.absolute():
            raise LineageError("lineage manifest self-reference is forbidden")
        result[member.label] = path
    return result


def hash_file(path: Path, label: str) -> tuple[str, int]:
    """Hash one stable regular inode through a no-follow descriptor."""
    value = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise LineageError(f"lineage member is not a regular file: {label}")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise LineageError(f"cannot hash lineage member: {label}") from exc
    stable_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    stable_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if stable_before != stable_after:
        raise LineageError(f"lineage member changed while hashing: {label}")
    return value.hexdigest(), after.st_size
