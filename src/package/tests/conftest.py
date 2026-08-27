from __future__ import annotations

from collections.abc import Iterator
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from so101_pusht_benchmark.workspace import runtime_artifact_root


def _canonical_receipt_binding() -> Path:
    artifact_root = runtime_artifact_root().resolve()
    binding = hashlib.sha256(str(artifact_root).encode()).hexdigest()
    return Path.home() / ".local/state/so101-pusht-benchmark/producer-receipts" / binding


def _tree_snapshot(path: Path) -> tuple[tuple[str, str, str], ...]:
    if not path.exists() and not path.is_symlink():
        return ()
    entries = [path, *sorted(path.rglob("*"))] if path.is_dir() else [path]
    snapshot: list[tuple[str, str, str]] = []
    for entry in entries:
        relative = "." if entry == path else entry.relative_to(path).as_posix()
        if entry.is_symlink():
            snapshot.append((relative, "symlink", entry.readlink().as_posix()))
        elif entry.is_file():
            snapshot.append((relative, "file", hashlib.sha256(entry.read_bytes()).hexdigest()))
        elif entry.is_dir():
            snapshot.append((relative, "directory", ""))
        else:
            snapshot.append((relative, "other", ""))
    return tuple(snapshot)


def canonical_final_state_snapshot() -> dict[str, tuple[tuple[str, str, str], ...]]:
    """Hash final production state and its owner-authenticated receipt binding."""
    artifact_root = runtime_artifact_root().resolve()
    return {
        "artifact-index.json": _tree_snapshot(artifact_root / "artifact-index.json"),
        "models": _tree_snapshot(artifact_root / "models"),
        "evaluations": _tree_snapshot(artifact_root / "evaluations"),
        "producer-receipts": _tree_snapshot(_canonical_receipt_binding()),
    }


def _assert_canonical_state_unchanged(
    before: dict[str, tuple[tuple[str, str, str], ...]], label: str
) -> None:
    after = canonical_final_state_snapshot()
    assert after == before, f"{label} mutated canonical final production state"


@pytest.fixture(scope="session", autouse=True)
def canonical_final_state_unchanged_by_full_suite() -> Iterator[None]:
    """Prove the complete pytest run leaves canonical final state byte-identical."""
    before = canonical_final_state_snapshot()
    yield
    _assert_canonical_state_unchanged(before, "full pytest suite")


@pytest.fixture(autouse=True)
def canonical_final_state_unchanged_by_test() -> Iterator[None]:
    """Attribute any canonical final-state leak to the exact test that caused it."""
    before = canonical_final_state_snapshot()
    yield
    _assert_canonical_state_unchanged(before, "test")


@pytest.fixture(autouse=True)
def isolated_producer_receipts(secure_test_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route every in-process producer receipt to one per-test owner-only home."""
    from so101_pusht_benchmark.training import artifacts

    class TestAccount:
        pw_dir = str(secure_test_home)

    def account_lookup(_uid: int) -> TestAccount:
        return TestAccount()

    monkeypatch.setattr(artifacts.pwd, "getpwuid", account_lookup)


@pytest.fixture
def canonical_test_root() -> Iterator[Path]:
    """Give one test exclusive storage below the canonical artifact root."""
    artifact_root = runtime_artifact_root()
    artifact_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"pytest-{os.getpid()}-", dir=artifact_root) as temporary:
        yield Path(temporary)


@pytest.fixture
def secure_test_home() -> Iterator[Path]:
    """Give producer-receipt tests an owner-only account-home fixture."""
    state_root = Path.home() / ".local/state"
    with TemporaryDirectory(prefix=f"pytest-home-{os.getpid()}-", dir=state_root) as temporary:
        yield Path(temporary)
