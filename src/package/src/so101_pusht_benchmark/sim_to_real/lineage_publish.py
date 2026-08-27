"""Pinned no-follow atomic authority receipt publication."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat

from .lineage_types import LineageError


@dataclass(slots=True)
class OutputTarget:
    """Pinned output parent and protected inode authority."""

    directory_fd: int
    name: str
    parent: Path
    parent_identity: tuple[int, int]
    protected_inodes: set[tuple[int, int]]

    def close(self) -> None:
        """Release the pinned directory descriptor."""
        if self.directory_fd >= 0:
            os.close(self.directory_fd)
            self.directory_fd = -1


def _open_parent(parent: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    absolute = parent.absolute()
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _inode(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _destination_info(target: OutputTarget) -> os.stat_result | None:
    try:
        return os.stat(target.name, dir_fd=target.directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def prepare_output(output: Path, protected: tuple[Path, ...]) -> OutputTarget:
    """Verify output parents and aliases before removing an old receipt."""
    destination = output.absolute()
    if destination.name in {"", ".", ".."}:
        raise LineageError("receipt output must name a file")
    try:
        parent_fd = _open_parent(destination.parent)
    except OSError as exc:
        raise LineageError("receipt output parent is missing, non-directory, or symlinked") from exc
    protected_inodes = {_inode(path.lstat()) for path in protected}
    target = OutputTarget(
        parent_fd,
        destination.name,
        destination.parent,
        _inode(os.fstat(parent_fd)),
        protected_inodes,
    )
    try:
        existing = _destination_info(target)
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                raise LineageError("receipt output cannot be a symlink")
            if not stat.S_ISREG(existing.st_mode):
                raise LineageError("receipt output must be a regular file")
            if _inode(existing) in protected_inodes:
                raise LineageError("receipt output aliases an immutable lineage member")
            os.unlink(target.name, dir_fd=target.directory_fd)
    except BaseException:
        target.close()
        raise
    else:
        return target


def _verify_lexical_parent(target: OutputTarget) -> int:
    try:
        descriptor = _open_parent(target.parent)
    except OSError as exc:
        raise LineageError("receipt output parent identity changed") from exc
    if _inode(os.fstat(descriptor)) != target.parent_identity:
        os.close(descriptor)
        raise LineageError("receipt output parent identity changed")
    return descriptor


def lexical_parent_verifier() -> Callable[[OutputTarget], int]:
    """Expose the final-check hook for adversarial race regression injection."""
    return _verify_lexical_parent


def _require_destination_absent(target: OutputTarget) -> None:
    info = _destination_info(target)
    if info is None:
        return
    if stat.S_ISLNK(info.st_mode) or _inode(info) in target.protected_inodes:
        raise LineageError("receipt output destination alias changed after preparation")
    raise LineageError("receipt output destination changed after preparation")


def _verify_published(target: OutputTarget, content: bytes) -> tuple[int, int]:
    lexical_fd = _verify_lexical_parent(target)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        pinned_info = os.stat(target.name, dir_fd=target.directory_fd, follow_symlinks=False)
        descriptor = os.open(target.name, flags, dir_fd=lexical_fd)
        with os.fdopen(descriptor, "rb") as stream:
            lexical_info = os.fstat(stream.fileno())
            digest = hashlib.sha256(stream.read()).digest()
    except (FileNotFoundError, OSError) as exc:
        raise LineageError("published receipt is absent or unsafe at requested path") from exc
    finally:
        os.close(lexical_fd)
    if (
        not stat.S_ISREG(pinned_info.st_mode)
        or _inode(pinned_info) != _inode(lexical_info)
        or digest != hashlib.sha256(content).digest()
    ):
        raise LineageError("published receipt identity or digest mismatch")
    return _inode(pinned_info)


def _direct_parent_identity(parent: Path) -> tuple[int, int]:
    current = Path(parent.absolute().anchor)
    info = current.lstat()
    for part in parent.absolute().parts[1:]:
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise LineageError("receipt output parent identity changed")
    return _inode(info)


def _verify_requested_path_direct(
    target: OutputTarget, published_inode: tuple[int, int], content: bytes
) -> None:
    requested = target.parent / target.name
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        if _direct_parent_identity(target.parent) != target.parent_identity:
            raise LineageError("receipt output parent identity changed")
        descriptor = os.open(requested, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            published_digest = hashlib.sha256(stream.read()).digest()
        final_name = requested.lstat()
        final_parent = _direct_parent_identity(target.parent)
    except (FileNotFoundError, OSError) as exc:
        raise LineageError("published receipt is absent at requested lexical path") from exc
    if (
        final_parent != target.parent_identity
        or not stat.S_ISREG(opened.st_mode)
        or _inode(opened) != published_inode
        or opened.st_size != len(content)
        or published_digest != hashlib.sha256(content).digest()
        or _inode(final_name) != published_inode
        or final_name.st_size != len(content)
    ):
        raise LineageError("published receipt lexical identity or digest mismatch")


def _unlink_if_task_inode(target: OutputTarget, name: str, task_inode: tuple[int, int]) -> None:
    try:
        info = os.stat(name, dir_fd=target.directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if _inode(info) == task_inode:
        os.unlink(name, dir_fd=target.directory_fd)


def publish_output(target: OutputTarget, content: bytes) -> None:
    """Detect mutations through the final lexical check; later same-user changes are external."""
    temporary = f".{target.name}.{os.getpid()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    task_inode: tuple[int, int] | None = None
    try:
        parent_check = _verify_lexical_parent(target)
        os.close(parent_check)
        _require_destination_absent(target)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=target.directory_fd)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        task_inode = _inode(os.stat(temporary, dir_fd=target.directory_fd, follow_symlinks=False))
        parent_check = _verify_lexical_parent(target)
        os.close(parent_check)
        _require_destination_absent(target)
        os.replace(
            temporary,
            target.name,
            src_dir_fd=target.directory_fd,
            dst_dir_fd=target.directory_fd,
        )
        os.fsync(target.directory_fd)
        published_inode = _verify_published(target, content)
        _verify_requested_path_direct(target, published_inode, content)
    except BaseException:
        if task_inode is not None:
            _unlink_if_task_inode(target, temporary, task_inode)
            _unlink_if_task_inode(target, target.name, task_inode)
        raise
