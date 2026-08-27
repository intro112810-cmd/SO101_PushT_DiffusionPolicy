"""Strict dirfd-based durable writes for guarded rollout evidence."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat

from .rollout_codes import RolloutCode, RolloutViolation

__all__ = (
    "ExclusiveAppendFile",
    "LeafIdentity",
    "atomic_write_new",
    "read_regular_leaf",
    "unlink_owned_leaf",
)

Identity = tuple[int, int]


def _identity(info: os.stat_result) -> Identity:
    return info.st_dev, info.st_ino


def _leaf_name(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or os.sep in name:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "unsafe evidence leaf name")
    return name


def _open_directory(path: Path) -> int:
    if ".." in path.parts:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "evidence path traversal")
    absolute = path.absolute()
    if ".." in absolute.parts:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "evidence path traversal")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(absolute.anchor, flags)
        for part in absolute.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise RolloutViolation(
            RolloutCode.R_HASH_MISMATCH,
            "evidence parent is missing, replaced, non-directory, or symlinked",
        ) from exc
    return descriptor


@dataclass(frozen=True, slots=True)
class LeafIdentity:
    """Identity of a leaf created under one pinned directory."""

    directory: Path
    directory_identity: Identity
    name: str
    inode: Identity
    owned: bool


class _PinnedDirectory:
    def __init__(self, path: Path) -> None:
        self.path = path.absolute()
        self.fd = _open_directory(self.path)
        self.identity = _identity(os.fstat(self.fd))

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def verify(self) -> None:
        check = _open_directory(self.path)
        try:
            if _identity(os.fstat(check)) != self.identity:
                raise RolloutViolation(
                    RolloutCode.R_HASH_MISMATCH, "evidence parent identity changed"
                )
        finally:
            os.close(check)

    def stat(self, name: str) -> os.stat_result | None:
        try:
            return os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def require_regular(self, name: str, expected: Identity | None = None) -> os.stat_result:
        info = self.stat(name)
        if info is None or not stat.S_ISREG(info.st_mode):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "evidence leaf is not regular")
        if expected is not None and _identity(info) != expected:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "evidence leaf identity changed")
        return info

    def unlink_if(self, name: str, expected: Identity) -> None:
        info = self.stat(name)
        if info is not None and _identity(info) == expected:
            os.unlink(name, dir_fd=self.fd)


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short evidence write")
        remaining = remaining[written:]


def read_regular_leaf(directory: Path, name: str) -> tuple[bytes, os.stat_result]:
    """Read one regular non-symlink leaf while pinning its parent and inode."""
    leaf = _leaf_name(name)
    pinned = _PinnedDirectory(directory)
    descriptor = -1
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW
        descriptor = os.open(leaf, flags, dir_fd=pinned.fd)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "evidence leaf is not regular")
        pinned.require_regular(leaf, _identity(opened))
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        pinned.require_regular(leaf, _identity(opened))
        pinned.verify()
        return b"".join(chunks), opened
    except OSError as exc:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "unsafe evidence leaf") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        pinned.close()


def atomic_write_new(
    directory: Path,
    name: str,
    content: bytes,
    *,
    temporary: str,
    accept_identical: bool = False,
) -> LeafIdentity:
    """Fsync and atomically publish a new leaf through one pinned directory fd."""
    leaf = _leaf_name(name)
    temp = _leaf_name(temporary)
    if leaf == temp:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "temporary leaf aliases output")
    pinned = _PinnedDirectory(directory)
    created: Identity | None = None
    published = False
    try:
        existing = pinned.stat(leaf)
        if existing is not None:
            if not accept_identical:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "evidence leaf already exists")
            prior, prior_info = read_regular_leaf(directory, leaf)
            if prior != content:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "evidence content changed")
            pinned.require_regular(leaf, _identity(prior_info))
            pinned.verify()
            return LeafIdentity(directory, pinned.identity, leaf, _identity(prior_info), False)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(temp, flags, 0o600, dir_fd=pinned.fd)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "temporary leaf is not regular")
            created = _identity(opened)
            _write_all(descriptor, content)
            os.fsync(descriptor)
            pinned.require_regular(temp, created)
            pinned.verify()
            if pinned.stat(leaf) is not None:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "evidence leaf appeared")
            os.link(
                temp,
                leaf,
                src_dir_fd=pinned.fd,
                dst_dir_fd=pinned.fd,
                follow_symlinks=False,
            )
            pinned.require_regular(leaf, created)
            os.unlink(temp, dir_fd=pinned.fd)
            os.fsync(pinned.fd)
            pinned.require_regular(leaf, created)
            pinned.verify()
            published = True
            return LeafIdentity(directory, pinned.identity, leaf, created, True)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "unsafe evidence publication") from exc
    finally:
        if created is not None and not published:
            pinned.unlink_if(temp, created)
            pinned.unlink_if(leaf, created)
        pinned.close()


def unlink_owned_leaf(identity: LeafIdentity) -> None:
    """Remove only a leaf still carrying the inode created by this process."""
    if not identity.owned:
        return
    pinned = _PinnedDirectory(identity.directory)
    try:
        if pinned.identity != identity.directory_identity:
            return
        pinned.unlink_if(identity.name, identity.inode)
        os.fsync(pinned.fd)
    finally:
        pinned.close()


class ExclusiveAppendFile:
    """Append to one exclusively created regular leaf with stable inode identity."""

    def __init__(self, path: Path) -> None:
        self._directory = path.parent
        self._name = _leaf_name(path.name)
        self._directory_identity: Identity | None = None
        self._inode: Identity | None = None

    def append(self, content: bytes) -> None:
        """Fsync one append and reject any parent or leaf replacement."""
        pinned = _PinnedDirectory(self._directory)
        created = False
        descriptor = -1
        opened_inode: Identity | None = None
        completed = False
        try:
            flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW
            if self._inode is None:
                flags |= os.O_CREAT | os.O_EXCL
                created = True
            elif pinned.identity != self._directory_identity:
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "append parent changed")
            descriptor = os.open(self._name, flags, 0o600, dir_fd=pinned.fd)
            opened = os.fstat(descriptor)
            inode = _identity(opened)
            opened_inode = inode
            if not stat.S_ISREG(opened.st_mode) or (
                self._inode is not None and inode != self._inode
            ):
                raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "append leaf changed")
            pinned.require_regular(self._name, inode)
            pinned.verify()
            _write_all(descriptor, content)
            os.fsync(descriptor)
            if created:
                os.fsync(pinned.fd)
            pinned.require_regular(self._name, inode)
            pinned.verify()
            self._directory_identity = pinned.identity
            self._inode = inode
            completed = True
        except OSError as exc:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "unsafe evidence append") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if created and not completed and self._inode is None and opened_inode is not None:
                pinned.unlink_if(self._name, opened_inode)
            pinned.close()
