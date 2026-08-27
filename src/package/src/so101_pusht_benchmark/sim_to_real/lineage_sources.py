"""Safe AST-derived Python source closure for the frozen replay route."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path

from .lineage_types import DEFAULT_ROOTS, LineageError, LineageRoots, Scope

_ROUTE_ENTRIES = (
    "run_recovered_checkpoint_rollout",
    "generate_feedback_artifacts",
    "capture_sim_to_real_samples",
    "audit_joint_equivalence_read_only",
    "audit_camera_registration",
    "so101_pusht_benchmark.sim_to_real.live_capture_provider",
    "so101_pusht_benchmark.sim_to_real.replay_receipts",
    "so101_pusht_benchmark.sim_to_real.lineage",
    "so101_pusht_benchmark.native_runtime",
    "so101_pusht_benchmark.integrations.paper_baselines.runner",
    "so101_pusht_benchmark.evaluation.frozen_env",
    "so101_pusht_benchmark.core.upstream_provenance",
)
_POLICY_MODULE = "diffusion_policy.policy.diffusion_unet_hybrid_image_policy"
_WORKSPACE_MODULE = "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace"


@dataclass(frozen=True, slots=True)
class SourceRoot:
    """One import namespace mapped to a pinned source directory."""

    prefix: str
    directory: Path
    scope: Scope
    manifest_prefix: str


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One AST-reachable Python source file."""

    module: str
    scope: Scope
    path: str
    source: Path


def _source_roots(roots: LineageRoots) -> tuple[SourceRoot, ...]:
    package_module = roots.package / "src/so101_pusht_benchmark"
    stanford = roots.project / "05_references/external_repos/real-stanford_diffusion_policy"
    return (
        SourceRoot("", roots.package / "scripts", "package", "scripts"),
        SourceRoot(
            "so101_pusht_benchmark",
            package_module,
            "package",
            "src/so101_pusht_benchmark",
        ),
        SourceRoot(
            "diffusion_policy",
            stanford / "diffusion_policy",
            "project",
            "05_references/external_repos/real-stanford_diffusion_policy/diffusion_policy",
        ),
        SourceRoot("robomimic", roots.runtime / "robomimic", "runtime", "robomimic"),
    )


def _module_relative(root: SourceRoot, module: str) -> str | None:
    if root.prefix:
        if module == root.prefix:
            return ""
        prefix = f"{root.prefix}."
        if not module.startswith(prefix):
            return None
        return module.removeprefix(prefix).replace(".", "/")
    if "." in module:
        return None
    return module


def _resolve_module(module: str, roots: tuple[SourceRoot, ...]) -> SourceRecord | None:
    for root in roots:
        relative = _module_relative(root, module)
        if relative is None:
            continue
        candidates = (
            root.directory / f"{relative}.py" if relative else root.directory / "__init__.py",
            root.directory / relative / "__init__.py",
        )
        for candidate in candidates:
            if candidate.is_file() and not candidate.is_symlink():
                path = candidate.relative_to(root.directory).as_posix()
                manifest_path = f"{root.manifest_prefix}/{path}"
                return SourceRecord(module, root.scope, manifest_path, candidate)
    return None


def _absolute_import(module: str, package: str, level: int) -> str:
    if level == 0:
        return module
    request = f"{'.' * level}{module}"
    try:
        return importlib.util.resolve_name(request, package)
    except ImportError as exc:
        raise LineageError(f"invalid relative import in consumed source: {request}") from exc


def _imports(record: SourceRecord) -> tuple[str, ...]:
    try:
        tree = ast.parse(record.source.read_bytes(), filename=str(record.source))
    except (OSError, SyntaxError) as exc:
        raise LineageError(f"cannot parse consumed source: {record.path}") from exc
    package = (
        record.module if record.source.name == "__init__.py" else record.module.rpartition(".")[0]
    )
    discovered: set[str] = set()

    class ImportVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            discovered.update(alias.name for alias in node.names)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            base = _absolute_import(node.module or "", package, node.level)
            if base:
                discovered.add(base)
            discovered.update(f"{base}.{alias.name}" for alias in node.names if base)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            del node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            del node

        def visit_Lambda(self, node: ast.Lambda) -> None:
            del node

    ImportVisitor().visit(tree)
    return tuple(sorted(discovered))


def _package_initializers(module: str, roots: tuple[SourceRoot, ...]) -> tuple[str, ...]:
    parts = module.split(".")
    result: list[str] = []
    for index in range(1, len(parts)):
        parent = ".".join(parts[:index])
        record = _resolve_module(parent, roots)
        if record is not None and record.source.name == "__init__.py":
            result.append(parent)
    return tuple(result)


def derive_route_closure(roots: LineageRoots) -> tuple[SourceRecord, ...]:
    """Derive explicit imports plus every implicitly loaded package initializer."""
    source_roots = _source_roots(roots)
    pending = [*_ROUTE_ENTRIES, _POLICY_MODULE, _WORKSPACE_MODULE]
    records: dict[str, SourceRecord] = {}
    while pending:
        module = pending.pop(0)
        if module in records:
            continue
        pending.extend(
            parent
            for parent in _package_initializers(module, source_roots)
            if parent not in records
        )
        record = _resolve_module(module, source_roots)
        if record is None:
            if module in {*_ROUTE_ENTRIES, _POLICY_MODULE, _WORKSPACE_MODULE}:
                raise LineageError(f"consumed route source is missing: {module}")
            continue
        records[module] = record
        pending.extend(
            imported
            for imported in _imports(record)
            if imported not in records and _resolve_module(imported, source_roots) is not None
        )
    return tuple(sorted(records.values(), key=lambda item: (item.scope, item.path)))


def _require_origin(module: str, expected: Path) -> None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError) as exc:
        raise LineageError(f"installed consumed source origin is unavailable: {module}") from exc
    if spec is None or spec.origin is None:
        raise LineageError(f"installed consumed source origin is unavailable: {module}")
    origin = Path(spec.origin)
    if origin.is_symlink() or origin.absolute() != expected.absolute():
        raise LineageError(f"installed consumed source origin escapes pinned root: {module}")


def _require_search_root(module: str, expected: Path) -> None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError) as exc:
        raise LineageError(f"installed package search root is unavailable: {module}") from exc
    locations = (
        ()
        if spec is None or spec.submodule_search_locations is None
        else tuple(spec.submodule_search_locations)
    )
    if len(locations) != 1 or Path(locations[0]).absolute() != expected.absolute():
        raise LineageError(f"installed package search root escapes pinned root: {module}")


def validate_installed_origins(roots: LineageRoots) -> None:
    """Bind actual import selection to the pinned project, Stanford, and runtime roots."""
    if roots != DEFAULT_ROOTS:
        return
    project_package = roots.package / "src/so101_pusht_benchmark"
    _require_origin("so101_pusht_benchmark", project_package / "__init__.py")
    _require_search_root("so101_pusht_benchmark", project_package)
    stanford = roots.project / "05_references/external_repos/real-stanford_diffusion_policy"
    _require_search_root("diffusion_policy", stanford / "diffusion_policy")
    _require_origin(
        _POLICY_MODULE,
        stanford / "diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py",
    )
    _require_origin(
        _WORKSPACE_MODULE,
        stanford / "diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py",
    )
    _require_origin("robomimic", roots.runtime / "robomimic/__init__.py")
    _require_search_root("robomimic", roots.runtime / "robomimic")
    _require_origin("scservo_sdk", roots.runtime / "scservo_sdk/__init__.py")
    _require_search_root("scservo_sdk", roots.runtime / "scservo_sdk")


def source_inventory(roots: LineageRoots) -> set[tuple[Scope, str]]:
    """Return the independently derived scope/path inventory."""
    return {(record.scope, record.path) for record in derive_route_closure(roots)}
