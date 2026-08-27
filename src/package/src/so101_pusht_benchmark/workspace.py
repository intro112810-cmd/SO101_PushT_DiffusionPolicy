"""Machine-readable workspace boundary and artifact-routing policy."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import TypedDict, cast

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
STATUS_CONFIG = PACKAGE_ROOT / "configs/workspace_status.yaml"


class WorkspacePolicyError(ValueError):
    """Raised when the workspace policy is malformed or unsafe."""


class _Prototype(TypedDict):
    status: str
    runtime_import: str
    path: str


class _Workspace(TypedDict):
    status: str
    mode: str
    artifact_root: str
    path: str
    plan: str
    legacy_modes_superseded: list[str]


class _NativeField(TypedDict, total=False):
    key: str
    dtype: str
    shape: list[int]
    order: list[str]
    meaning: str
    bounds: list[float]


class _NativeContract(TypedDict):
    schema: str
    images: dict[str, _NativeField]
    state: _NativeField
    action: _NativeField
    fps: int


class _Runtime(TypedDict):
    native_lock: str
    native_lock_sha256: str
    fallback: str
    generated_artifacts_root: str


class _Historical(TypedDict):
    plans: list[str]
    configs: list[str]


class _IdentityScopes(TypedDict):
    training: str
    evaluation: str
    bundle: str


class _ModelAuthority(TypedDict):
    identities: dict[str, _IdentityScopes]
    hardware_control: str


class _RealDiagnosticRollout(TypedDict):
    status: str
    authority: str
    deployment_scope: str
    plan: str
    hardware_profile: str
    module_root: str
    allowed_scripts: list[str]
    require_owner_approved_policy: bool
    require_single_owner_writer: bool


class _Routing(TypedDict):
    runtime_artifacts_root: str
    obsidian_exception_requires_user_request: bool
    no_automatic_obsidian_write: bool
    obsidian_markdown_exception: str


class _Archive(TypedDict):
    physical_archive: str


class WorkspacePolicy(TypedDict):
    schema: int
    workspace: _Workspace
    native_contract: _NativeContract
    active_configs: dict[str, str]
    model_authority: _ModelAuthority
    real_diagnostic_rollout: _RealDiagnosticRollout
    historical: _Historical
    prototypes: list[_Prototype]
    runtime: _Runtime
    report_routing: _Routing
    archive: _Archive


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise WorkspacePolicyError(f"{label} must be a mapping")
    return cast("dict[str, object]", value)


def _validate_workspace(raw: dict[str, object]) -> _Workspace:
    workspace = _mapping(raw.get("workspace"), "workspace")
    if workspace.get("status") != "active":
        raise WorkspacePolicyError("active workspace status is required")
    if workspace.get("mode") != "native_pusht_so100_four_model_benchmark":
        raise WorkspacePolicyError("unexpected active workspace mode")
    if workspace.get("plan") != ".omo/plans/pusht-so100-four-model-clean-restart.md":
        raise WorkspacePolicyError("clean-restart must be the sole governing plan")
    if "base_plan" in workspace or "plans" in workspace:
        raise WorkspacePolicyError("clean-restart must be the sole governing plan")
    return cast("_Workspace", cast("object", workspace))


def _validate_native_contract(raw: dict[str, object]) -> _NativeContract:
    contract = _mapping(raw.get("native_contract"), "native_contract")
    expected: dict[str, object] = {
        "schema": "pusht-so100-native-v1",
        "images": {
            "cam_top": {"dtype": "uint8", "shape": [224, 224, 3]},
            "cam_side": {"dtype": "uint8", "shape": [224, 224, 3]},
        },
        "state": {
            "key": "agent_pos",
            "dtype": "float32",
            "shape": [5],
            "order": ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        },
        "action": {
            "dtype": "float32",
            "shape": [2],
            "meaning": "absolute_mocap_xy",
            "bounds": [-1.0, 1.0],
        },
        "fps": 10,
    }
    if contract != expected:
        raise WorkspacePolicyError("native pushT-so100 contract mismatch")
    return cast("_NativeContract", cast("object", contract))


def _validate_config_routes(raw: dict[str, object]) -> tuple[dict[str, str], _Historical]:
    active = _mapping(raw.get("active_configs"), "active_configs")
    historical = _mapping(raw.get("historical"), "historical")
    if set(active) != {"benchmark", "collection", "export"} or not all(
        isinstance(value, str) for value in active.values()
    ):
        raise WorkspacePolicyError("active native config routes are malformed")
    historical_configs = historical.get("configs")
    historical_plans = historical.get("plans")
    if not isinstance(historical_configs, list) or not isinstance(historical_plans, list):
        raise WorkspacePolicyError("historical routes are malformed")
    if set(cast("dict[str, str]", cast("object", active)).values()) & set(
        cast("list[str]", historical_configs)
    ):
        raise WorkspacePolicyError("active and historical config routes must be disjoint")
    return (
        cast("dict[str, str]", cast("object", active)),
        cast("_Historical", cast("object", historical)),
    )


def _validate_model_authority(raw: dict[str, object]) -> _ModelAuthority:
    authority = _mapping(raw.get("model_authority"), "model authority")
    identities = _mapping(authority.get("identities"), "model identities")
    expected_models = {"dp_cnn", "dp_transformer", "ibc", "lstm_gmm"}
    expected_scopes = {
        "training": "simulation_only",
        "evaluation": "simulation_only",
        "bundle": "simulation_only",
    }
    if (
        set(identities) != expected_models
        or authority.get("hardware_control") != "forbidden"
        or set(authority) != {"identities", "hardware_control"}
    ):
        raise WorkspacePolicyError("simulation identity cannot control hardware")
    for value in identities.values():
        scopes = _mapping(value, "model identity scopes")
        if scopes != expected_scopes:
            raise WorkspacePolicyError("simulation identity cannot control hardware")
    return cast("_ModelAuthority", cast("object", authority))


def _validate_real_diagnostic_rollout(
    raw: dict[str, object],
) -> _RealDiagnosticRollout:
    route = _mapping(raw.get("real_diagnostic_rollout"), "diagnostic rollout governance")
    expected: dict[str, object] = {
        "status": "governed",
        "authority": "separate_control_plane",
        "deployment_scope": "physical_diagnostic_only",
        "plan": ".omo/plans/sim-to-real-first-rollout.md",
        "hardware_profile": "configs/hardware/so101_real_v1.yaml",
        "module_root": "so101_pusht_benchmark.sim_to_real",
        "allowed_scripts": [
            "scripts/check_guarded_single_step.py",
            "scripts/run_guarded_single_step.py",
            "scripts/run_guarded_bounded_rollout.py",
            "scripts/verify_guarded_rollout.py",
        ],
        "require_owner_approved_policy": True,
        "require_single_owner_writer": True,
    }
    if route != expected:
        raise WorkspacePolicyError("diagnostic rollout governance is malformed")
    return cast("_RealDiagnosticRollout", cast("object", route))


def _validate_prototypes(raw: dict[str, object]) -> list[_Prototype]:
    prototypes = raw.get("prototypes")
    if not isinstance(prototypes, list) or not prototypes:
        raise WorkspacePolicyError("frozen prototypes are required")
    result: list[_Prototype] = []
    for value in cast("list[object]", prototypes):
        prototype = _mapping(value, "prototype")
        if prototype.get("status") != "frozen":
            raise WorkspacePolicyError("all prototype entries must be frozen")
        if prototype.get("runtime_import") != "forbidden":
            raise WorkspacePolicyError("frozen prototype imports must be forbidden")
        result.append(cast("_Prototype", cast("object", prototype)))
    return result


def _validate_routing(raw: dict[str, object], workspace: _Workspace) -> _Routing:
    runtime = cast("_Runtime", cast("object", _mapping(raw.get("runtime"), "runtime")))
    routing = cast(
        "_Routing", cast("object", _mapping(raw.get("report_routing"), "report_routing"))
    )
    if runtime.get("native_lock") != "environments/sim-runtime.lock":
        raise WorkspacePolicyError("native runtime lock route mismatch")
    digest = runtime.get("native_lock_sha256")
    if len(digest) != 64:
        raise WorkspacePolicyError("native runtime lock digest is malformed")
    if runtime.get("fallback") != "forbidden":
        raise WorkspacePolicyError("native runtime fallback must be forbidden")
    if runtime.get("generated_artifacts_root") != workspace.get("artifact_root"):
        raise WorkspacePolicyError("runtime artifact root must match workspace artifact root")
    if routing.get("runtime_artifacts_root") != workspace.get("artifact_root"):
        raise WorkspacePolicyError("report runtime root must match workspace artifact root")
    if routing.get("obsidian_exception_requires_user_request") is not True:
        raise WorkspacePolicyError("Obsidian exception must require an explicit request")
    if routing.get("no_automatic_obsidian_write") is not True:
        raise WorkspacePolicyError("automatic Obsidian writes must be disabled")
    return routing


def load_workspace_policy(path: Path = STATUS_CONFIG) -> WorkspacePolicy:
    """Load and validate the machine-consumed workspace policy."""
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WorkspacePolicyError(f"cannot load workspace policy: {path}") from exc
    raw = _mapping(value, "workspace policy")
    if raw.get("schema") != 2:
        raise WorkspacePolicyError("workspace policy schema must be 2")
    workspace = _validate_workspace(raw)
    _validate_native_contract(raw)
    _validate_config_routes(raw)
    _validate_model_authority(raw)
    _validate_real_diagnostic_rollout(raw)
    _validate_prototypes(raw)
    _validate_routing(raw, workspace)
    return cast("WorkspacePolicy", cast("object", raw))


def authorize_real_diagnostic_route(
    policy: WorkspacePolicy,
    *,
    authority: str,
    entry_point: str,
) -> _RealDiagnosticRollout:
    """Authorize only an explicitly governed sim-to-real diagnostic entry point."""
    if authority in {"training", "evaluation", "evaluator", "bundle"}:
        raise WorkspacePolicyError("simulation identity cannot control hardware")
    route = policy["real_diagnostic_rollout"]
    if authority != route["authority"]:
        raise WorkspacePolicyError("hardware authority source is not governed")
    module_root = route["module_root"]
    module_allowed = entry_point == module_root or entry_point.startswith(f"{module_root}.")
    if not module_allowed and entry_point not in route["allowed_scripts"]:
        raise WorkspacePolicyError("entry point is outside guarded sim-to-real route")
    return route


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a policy path under the project root without allowing traversal.

    The generated-artifacts root may live on a dedicated data mount linked from
    the project root (e.g. ``04_experiments -> /data/df/.../04_experiments``).
    In that case the lexical project-root path is the policy path; the resolved
    real location is validated against the allowed artifact mounts instead of
    requiring it to sit under the project root.
    """
    lexical = PROJECT_ROOT / relative_path
    candidate = lexical.resolve()
    if candidate != PROJECT_ROOT and PROJECT_ROOT not in candidate.parents:
        allowed_mount = Path(os.environ.get("PUSHT_ARTIFACT_MOUNT", "/data/df/02_InTro_Project"))
        if not allowed_mount.is_absolute():
            raise WorkspacePolicyError(f"artifact mount must be absolute: {allowed_mount}")
        try:
            inside_mount = candidate != allowed_mount and allowed_mount in candidate.parents
        except OSError:
            inside_mount = False
        if not inside_mount:
            raise WorkspacePolicyError(f"path escapes project root: {relative_path}")
    return candidate


def runtime_artifact_root() -> Path:
    """Return the sole runtime artifact root."""
    return resolve_project_path(load_workspace_policy()["runtime"]["generated_artifacts_root"])


def _check_lexical_path(path: Path) -> None:
    """Reject symlink components and non-directory existing parents before resolve."""
    lexical = path if path.is_absolute() else Path.cwd() / path
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise WorkspacePolicyError(f"symlink path component is forbidden: {path}")
        if current != lexical and not stat.S_ISDIR(info.st_mode):
            raise WorkspacePolicyError(f"parent is not a regular directory: {path}")


def _require_safe_file_path(candidate: Path) -> None:
    """Require a regular existing target or a future file beneath regular parents."""
    try:
        target_info = candidate.lstat()
    except FileNotFoundError:
        target_info = None
    if target_info is not None and not stat.S_ISREG(target_info.st_mode):
        raise WorkspacePolicyError(f"report target is not a regular file: {candidate}")
    parent = candidate.parent
    while True:
        try:
            parent_info = parent.lstat()
        except FileNotFoundError as exc:
            raise WorkspacePolicyError(f"report parent does not exist: {candidate}") from exc
        if not stat.S_ISDIR(parent_info.st_mode):
            raise WorkspacePolicyError(f"report parent is not a directory: {candidate}")
        if parent == Path(parent.anchor):
            break
        parent = parent.parent


def validate_report_path(path: Path, *, user_requested_obsidian: bool = False) -> Path:
    """Validate a generated report destination against the routing policy."""
    expanded = path.expanduser()
    _check_lexical_path(expanded)
    candidate = expanded.resolve()
    artifact_root = runtime_artifact_root().resolve()
    routing = load_workspace_policy()["report_routing"]
    obsidian_root = Path(routing["obsidian_markdown_exception"]).resolve()
    is_artifact = candidate != artifact_root and artifact_root in candidate.parents
    is_obsidian = (
        user_requested_obsidian and candidate.parent == obsidian_root and candidate.suffix == ".md"
    )
    if not (is_artifact or is_obsidian):
        raise WorkspacePolicyError(f"report path is outside an allowed root: {path}")
    _require_safe_file_path(expanded.absolute())
    return candidate
