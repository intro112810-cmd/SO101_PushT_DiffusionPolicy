"""Identity-only export for validated native pushT-so100 persisted data."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from .paper_view import (
    PaperArray,
    PaperViewMetadata,
    require_sha256,
    trusted_runtime_lock_digest,
    write_paper_view,
)
from .paper_view_reader import load_paper_view


class ExportError(ValueError):
    """Raised when native evidence cannot be exported without changing it."""


def export_paper_view(
    dataset_root: Path,
    output_root: Path,
    *,
    runtime_lock_digest: str,
) -> Path:
    """Reload a native store and atomically export the exact same arrays and episodes."""
    loaded = load_paper_view(dataset_root)
    manifest = loaded.manifest
    source_lock = require_sha256(manifest.get("runtime_lock_digest"), "source runtime lock digest")
    requested_lock = require_sha256(runtime_lock_digest, "runtime lock digest")
    trusted_lock = trusted_runtime_lock_digest()
    if requested_lock != trusted_lock or source_lock != trusted_lock:
        raise ExportError("trusted runtime lock digest mismatch")
    canonical = require_sha256(manifest.get("canonical_digest"), "canonical digest")
    root_digest = require_sha256(manifest.get("root_digest"), "root digest")
    root_provenance_raw = manifest.get("root_provenance")
    if not isinstance(root_provenance_raw, dict):
        raise ExportError("root provenance is malformed")
    root_provenance = cast("dict[str, object]", root_provenance_raw)
    episode_ids_raw = manifest.get("episode_ids")
    if not isinstance(episode_ids_raw, list) or not all(
        isinstance(value, str) for value in cast("list[object]", episode_ids_raw)
    ):
        raise ExportError("episode IDs are malformed")
    records_raw = manifest.get("arrays")
    if not isinstance(records_raw, dict):
        raise ExportError("array records are malformed")
    records = cast("dict[str, object]", records_raw)
    arrays: dict[str, PaperArray] = {}
    for name, values in loaded.arrays.items():
        record = records.get(name)
        if not isinstance(record, dict):
            raise ExportError(f"array unit is malformed: {name}")
        typed = cast("dict[str, object]", record)
        if not isinstance(typed.get("unit"), str):
            raise ExportError(f"array unit is malformed: {name}")
        arrays[name] = PaperArray(values, cast(str, typed["unit"]))
    destination = output_root / "paper_view" / canonical
    try:
        return write_paper_view(
            destination,
            arrays,
            loaded.episode_ends,
            PaperViewMetadata(
                canonical,
                root_digest,
                root_provenance,
                cast("list[str]", episode_ids_raw),
                loaded.splits,
                requested_lock,
                manifest.get("training_eligible") is True,
            ),
        )
    except (OSError, ValueError) as exc:
        raise ExportError(f"native export failed: {exc}") from exc


def runtime_lock_digest(path: Path) -> str:
    from ..native_runtime import DEFAULT_LOCK

    if path.resolve() != DEFAULT_LOCK.resolve():
        raise ExportError("runtime lock path is not the trusted canonical lock")
    return trusted_runtime_lock_digest()
