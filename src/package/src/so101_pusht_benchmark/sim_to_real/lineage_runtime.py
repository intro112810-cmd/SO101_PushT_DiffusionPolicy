"""Fresh import trace and third-party runtime fingerprint authority."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from .lineage_io import object_mapping
from .lineage_types import LineageError, LineageRoots, Scope

BOUNDARY = (
    "python_sources_and_loaded_extensions_exact_third_party_distributions_"
    "version_origin_bound_runtime_lock_hashed"
)
_TRACE_MODULES = (
    "run_recovered_checkpoint_rollout",
    "generate_feedback_artifacts",
    "capture_sim_to_real_samples",
    "audit_joint_equivalence_read_only",
    "audit_camera_registration",
    "so101_pusht_benchmark.sim_to_real.live_capture_provider",
    "so101_pusht_benchmark.sim_to_real.replay_receipts",
    "so101_pusht_benchmark.sim_to_real.lineage",
    "so101_pusht_benchmark.native_runtime",
    "scservo_sdk",
    "diffusion_policy.policy.diffusion_unet_hybrid_image_policy",
    "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace",
)
_TRACE_MARKER = "@@SO101_LINEAGE_TRACE@@"
# Importing MuJoCo selects one of these presentation backends from environment
# variables. The frozen replay/inference route does not create a viewer or GL
# context, so neither backend is part of its consumed runtime distribution set.
_NON_CONSUMED_RENDER_BACKENDS = frozenset({"glfw", "pyopengl"})
_TRACE_PROGRAM = f"""import importlib,json,sys
before=set(sys.modules)
modules=json.loads(sys.argv[1])
for module in modules:
 importlib.import_module(module)
files={{name:str(getattr(value,"__file__")) for name,value in sys.modules.items() if name not in before and getattr(value,"__file__",None)}}
print("{_TRACE_MARKER}"+json.dumps(files,sort_keys=True))
"""


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    """Fresh observed source paths and canonical third-party fingerprint."""

    sources: set[tuple[Scope, str]]
    fingerprint: dict[str, object]


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
    except OSError as exc:
        raise LineageError(f"cannot hash traced runtime file: {path}") from exc
    return value.hexdigest()


def trace_modules(roots: LineageRoots, modules: tuple[str, ...]) -> dict[str, Path]:
    """Import the selected route in a fresh bounded subprocess and return new module files."""
    stanford = roots.project / "05_references/external_repos/real-stanford_diffusion_policy"
    python_path = os.pathsep.join(
        (
            str(roots.package / "src"),
            str(roots.package / "scripts"),
            str(stanford),
            str(roots.runtime),
        )
    )
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": python_path,
        "MPLBACKEND": "Agg",
    }
    try:
        process = subprocess.run(
            [sys.executable, "-c", _TRACE_PROGRAM, json.dumps(modules)],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
            env=environment,
            cwd=roots.package,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LineageError("fresh consumed-route import trace failed") from exc
    if process.returncode != 0:
        raise LineageError(f"fresh consumed-route import trace failed: {process.stderr[-500:]}")
    payload = next(
        (
            line.removeprefix(_TRACE_MARKER)
            for line in reversed(process.stdout.splitlines())
            if line.startswith(_TRACE_MARKER)
        ),
        None,
    )
    if payload is None:
        raise LineageError("fresh consumed-route import trace produced no inventory")
    try:
        parsed: object = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LineageError("fresh consumed-route import trace inventory is malformed") from exc
    result: dict[str, Path] = {}
    for module, raw_path in object_mapping(parsed, "fresh import trace").items():
        if not isinstance(raw_path, str):
            raise LineageError("fresh import trace path is malformed")
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise LineageError(f"fresh import trace origin is unsafe: {module}")
        result[module] = path.absolute()
    return result


def _source_path(path: Path, roots: LineageRoots) -> tuple[Scope, str] | None:
    candidates: tuple[tuple[Scope, Path], ...] = (
        ("package", roots.package),
        ("project", roots.project),
        ("runtime", roots.runtime),
    )
    for scope, root in candidates:
        try:
            relative = path.relative_to(root.absolute()).as_posix()
        except ValueError:
            continue
        if scope != "runtime" or relative.startswith("robomimic/"):
            return scope, relative
    return None


def observed_source_inventory(
    roots: LineageRoots, traced: dict[str, Path]
) -> set[tuple[Scope, str]]:
    """Normalize only exact-source trust domains from a fresh module trace."""
    result: set[tuple[Scope, str]] = set()
    for path in traced.values():
        source = _source_path(path, roots)
        if source is not None and path.suffix in {".py", ".pyw"}:
            result.add(source)
    return result


def _distribution_files() -> tuple[
    dict[Path, set[str]], dict[str, importlib.metadata.Distribution]
]:
    owners: dict[Path, set[str]] = {}
    distributions: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.metadata["Name"]
        except KeyError:
            continue
        canonical = name.lower().replace("_", "-")
        distributions[canonical] = distribution
        for item in distribution.files or ():
            path = Path(str(distribution.locate_file(item))).absolute()
            owners.setdefault(path, set()).add(canonical)
    return owners, distributions


def _metadata_record(name: str, distribution: importlib.metadata.Distribution) -> dict[str, object]:
    metadata_path: Path | None = None
    record_path: Path | None = None
    for item in distribution.files or ():
        value = Path(str(distribution.locate_file(item))).absolute()
        if item.name == "METADATA" and ".dist-info" in item.as_posix():
            metadata_path = value
        elif item.name == "RECORD" and ".dist-info" in item.as_posix():
            record_path = value
    if metadata_path is None:
        raise LineageError(f"distribution metadata is unavailable: {name}")
    root = Path(str(distribution.locate_file(""))).absolute()
    return {
        "name": name,
        "version": distribution.version,
        "root": str(root),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "record_path": None if record_path is None else str(record_path),
        "record_sha256": None if record_path is None else _sha256(record_path),
    }


def runtime_fingerprint(traced: dict[str, Path]) -> dict[str, object]:
    """Bind observed distributions and every newly loaded extension module byte."""
    owners, distributions = _distribution_files()
    names: set[str] = set()
    extension_modules: dict[Path, set[str]] = {}
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for module, path in traced.items():
        names.update(owners.get(path, set()) - _NON_CONSUMED_RENDER_BACKENDS)
        if path.name.endswith(suffixes):
            extension_modules.setdefault(path, set()).add(module)
    records = [_metadata_record(name, distributions[name]) for name in sorted(names)]
    extensions = [
        {
            "modules": sorted(modules),
            "path": str(path),
            "sha256": _sha256(path),
        }
        for path, modules in sorted(extension_modules.items(), key=lambda item: str(item[0]))
    ]
    return {
        "schema": "so101-third-party-runtime-fingerprint-v1",
        "distributions": records,
        "extensions": extensions,
    }


def observe_runtime(roots: LineageRoots) -> RuntimeObservation:
    """Produce the authoritative fresh route observation."""
    traced = trace_modules(roots, _TRACE_MODULES)
    return RuntimeObservation(observed_source_inventory(roots, traced), runtime_fingerprint(traced))


def validate_runtime_fingerprint(expected: object, observed: dict[str, object]) -> None:
    """Reject version, origin, metadata/RECORD, extension, or set drift."""
    declared = object_mapping(expected, "runtime fingerprint")
    if declared != observed:
        raise LineageError("third-party runtime fingerprint mismatch")
