from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from so101_pusht_benchmark.sim_to_real import lineage as lineage_module
from so101_pusht_benchmark.sim_to_real import lineage_publish, lineage_sources
from so101_pusht_benchmark.sim_to_real.lineage import (
    ArtifactAuthorityReceipt,
    LineageError,
    LineageRoots,
    validate_lineage,
    validate_lineage_to_file,
)
from so101_pusht_benchmark.sim_to_real.lineage_io import object_list, object_mapping, read_json
from so101_pusht_benchmark.sim_to_real.lineage_publish import OutputTarget
from so101_pusht_benchmark.sim_to_real.lineage_runtime import (
    observe_runtime,
    observed_source_inventory,
    runtime_fingerprint,
    trace_modules,
    validate_runtime_fingerprint,
)
from so101_pusht_benchmark.sim_to_real.lineage_sources import (
    derive_route_closure,
    source_inventory,
    validate_installed_origins,
)
from so101_pusht_benchmark.sim_to_real.lineage_types import DEFAULT_ROOTS

_ARTIFACT_ID = "local-dp_cnn-recovered-v3-seed0"
_STANFORD_COMMIT = "5ba07ac6661db573af695b419a7947ecb704690f"
_ROBOMIMIC_COMMIT = "62ed2de905caeb9133136e4d14d810a8b6baa96c"
_AUDIT_ROUTE_PATHS = (
    "scripts/audit_camera_registration.py",
    "scripts/audit_joint_equivalence_read_only.py",
    "src/so101_pusht_benchmark/__init__.py",
    "src/so101_pusht_benchmark/control/action_filter.py",
    "src/so101_pusht_benchmark/sim/__init__.py",
    "src/so101_pusht_benchmark/sim/dls_ik.py",
    "src/so101_pusht_benchmark/sim/env.py",
    "src/so101_pusht_benchmark/sim/safety.py",
    "src/so101_pusht_benchmark/sim/scene.py",
    "src/so101_pusht_benchmark/sim_to_real/__init__.py",
    "src/so101_pusht_benchmark/sim_to_real/camera_audit_cli.py",
    "src/so101_pusht_benchmark/sim_to_real/camera_authority.py",
    "src/so101_pusht_benchmark/sim_to_real/camera_corpus.py",
    "src/so101_pusht_benchmark/sim_to_real/camera_geometry.py",
    "src/so101_pusht_benchmark/sim_to_real/camera_registration.py",
    "src/so101_pusht_benchmark/sim_to_real/contracts.py",
    "src/so101_pusht_benchmark/sim_to_real/joint_equivalence.py",
    "src/so101_pusht_benchmark/sim_to_real/joint_equivalence_affine.py",
    "src/so101_pusht_benchmark/sim_to_real/joint_equivalence_authority.py",
    "src/so101_pusht_benchmark/sim_to_real/joint_equivalence_cli.py",
    "src/so101_pusht_benchmark/sim_to_real/joint_equivalence_corpus.py",
    "src/so101_pusht_benchmark/sim_to_real/joint_equivalence_fk.py",
    "src/so101_pusht_benchmark/sim_to_real/joint_mapping.py",
    "src/so101_pusht_benchmark/sim_to_real/physical_ik_fk.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_approval.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_canonical.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_io.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_parser.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_schema.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_types.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_values.py",
    "src/so101_pusht_benchmark/sim_to_real/receipt_routing.py",
    "src/so101_pusht_benchmark/sim_to_real/replay_receipts.py",
    "src/so101_pusht_benchmark/sim_to_real/replay_types.py",
    "src/so101_pusht_benchmark/sim_to_real/rollout_codes.py",
    "src/so101_pusht_benchmark/sim_to_real/task_frame.py",
    "src/so101_pusht_benchmark/sim_to_real/task_frame_bridge.py",
    "src/so101_pusht_benchmark/task/__init__.py",
    "src/so101_pusht_benchmark/task/metric.py",
    "src/so101_pusht_benchmark/task/spec.py",
    "src/so101_pusht_benchmark/task/validation.py",
)
_RUNTIME_PREFLIGHT_MEMBERS = {
    "project_metadata": "pyproject.toml",
    "uv_lock": "uv.lock",
    "runtime_lock": "environments/sim-runtime.lock",
    "runtime_lock_sha256": "environments/sim-runtime.lock.sha256",
    "workspace_status": "configs/workspace_status.yaml",
}
_PROCESS_CAPTURE_SOURCE_PATHS = (
    "src/so101_pusht_benchmark/sim_to_real/live_capture_acceptance.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_cleanup.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_failure.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_pair.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_process.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_protocol.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_readiness.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_supervisor.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_workers.py",
    "src/so101_pusht_benchmark/sim_to_real/read_only_authority_io.py",
    "src/so101_pusht_benchmark/sim_to_real/read_only_authority_types.py",
)
_CAPTURE_ROUTE_PATHS = (
    "scripts/capture_sim_to_real_samples.py",
    "src/so101_pusht_benchmark/__init__.py",
    "src/so101_pusht_benchmark/hardware_profile.py",
    "src/so101_pusht_benchmark/sim_to_real/__init__.py",
    "src/so101_pusht_benchmark/sim_to_real/contracts.py",
    "src/so101_pusht_benchmark/sim_to_real/fixture_sample_capture.py",
    "src/so101_pusht_benchmark/sim_to_real/joint_mapping.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_acceptance.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_adapters.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_cleanup.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_cli.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_failure.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_identity.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_pair.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_process.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_protocol.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_provider.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_readiness.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_supervisor.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_types.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_validation.py",
    "src/so101_pusht_benchmark/sim_to_real/live_capture_workers.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_approval.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_canonical.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_io.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_parser.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_schema.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_types.py",
    "src/so101_pusht_benchmark/sim_to_real/policy_values.py",
    "src/so101_pusht_benchmark/sim_to_real/readiness.py",
    "src/so101_pusht_benchmark/sim_to_real/read_only_authority.py",
    "src/so101_pusht_benchmark/sim_to_real/read_only_authority_io.py",
    "src/so101_pusht_benchmark/sim_to_real/read_only_authority_types.py",
    "src/so101_pusht_benchmark/sim_to_real/receipt_routing.py",
    "src/so101_pusht_benchmark/sim_to_real/rollout_codes.py",
    "src/so101_pusht_benchmark/sim_to_real/rollout_identity.py",
    "src/so101_pusht_benchmark/sim_to_real/rollout_record_types.py",
    "src/so101_pusht_benchmark/sim_to_real/sample_capture.py",
    "src/so101_pusht_benchmark/sim_to_real/secure_io.py",
)


@dataclass(frozen=True, slots=True)
class LineageFixture:
    artifact: Path
    package: Path
    project: Path
    runtime: Path
    authority: Path

    @property
    def roots(self) -> LineageRoots:
        return LineageRoots(self.package, self.project, self.runtime)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _identity(runtime_digest: str, environment_digest: str) -> dict[str, object]:
    return {
        "model": "dp_cnn",
        "policy_target": (
            "diffusion_policy.policy.diffusion_unet_hybrid_image_policy."
            "DiffusionUnetHybridImagePolicy"
        ),
        "policy_class": "DiffusionUnetHybridImagePolicy",
        "policy_module": "diffusion_policy.policy.diffusion_unet_hybrid_image_policy",
        "workspace_target": (
            "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace."
            "TrainDiffusionUnetHybridWorkspace"
        ),
        "workspace_class": "TrainDiffusionUnetHybridWorkspace",
        "workspace_module": "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace",
        "observation_steps": 2,
        "horizon": 16,
        "executed_actions": 8,
        "optimizer_updates": 400_000,
        "dataset_digest": "a" * 64,
        "split_digest": "b" * 64,
        "runtime_lock_digest": runtime_digest,
        "environment_manifest_digest": environment_digest,
        "stanford_commit": _STANFORD_COMMIT,
        "robomimic_commit": _ROBOMIMIC_COMMIT,
    }


def _write_source_graph(fixture: LineageFixture) -> None:
    files = {
        fixture.package / "src/so101_pusht_benchmark/__init__.py": b"VALUE = 1\n",
        fixture.package / "src/so101_pusht_benchmark/native_runtime.py": b"VALUE = 1\n",
        fixture.package / "src/so101_pusht_benchmark/core/__init__.py": b"VALUE = 1\n",
        fixture.package / "src/so101_pusht_benchmark/evaluation/__init__.py": b"VALUE = 1\n",
        fixture.package / "src/so101_pusht_benchmark/integrations/__init__.py": b"VALUE = 1\n",
        fixture.package
        / "src/so101_pusht_benchmark/integrations/paper_baselines/__init__.py": b"VALUE = 1\n",
        fixture.package
        / "src/so101_pusht_benchmark/sim_to_real/__init__.py": b"from . import contracts\n",
        fixture.package / "src/so101_pusht_benchmark/sim_to_real/contracts.py": b"VALUE = 1\n",
        fixture.package / "src/so101_pusht_benchmark/training/__init__.py": b"VALUE = 1\n",
        fixture.package / "scripts/run_recovered_checkpoint_rollout.py": (
            b"import generate_feedback_artifacts\n"
        ),
        fixture.package / "scripts/generate_feedback_artifacts.py": (
            b"from so101_pusht_benchmark.integrations.paper_baselines import runner\n"
        ),
        fixture.package / "scripts/verify_sim_to_real_lineage.py": (
            b"from so101_pusht_benchmark.sim_to_real import lineage\n"
        ),
        fixture.package
        / "src/so101_pusht_benchmark/integrations/paper_baselines/runner.py": b"VALUE = 1\n",
        fixture.package / "src/so101_pusht_benchmark/evaluation/frozen_env.py": b"VALUE = 1\n",
        fixture.package / "src/so101_pusht_benchmark/core/upstream_provenance.py": b"VALUE = 1\n",
        fixture.package / "src/so101_pusht_benchmark/sim_to_real/lineage.py": b"VALUE = 1\n",
        fixture.project / "05_references/external_repos/real-stanford_diffusion_policy/"
        "diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py": b"import robomimic\n",
        fixture.project / "05_references/external_repos/real-stanford_diffusion_policy/"
        "diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py": b"VALUE = 1\n",
        fixture.runtime / "robomimic/__init__.py": b"VALUE = 1\n",
        fixture.runtime / "scservo_sdk/__init__.py": b"VALUE = 1\n",
    }
    capture_imports: list[str] = []
    for relative in _CAPTURE_ROUTE_PATHS:
        path = fixture.package / relative
        files.setdefault(path, b"VALUE = 1\n")
        if relative.startswith("src/") and not relative.endswith("/__init__.py"):
            capture_imports.append(
                relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
            )
    files[fixture.package / "scripts/capture_sim_to_real_samples.py"] = (
        "\n".join(f"import {module}" for module in capture_imports) + "\n"
    ).encode()
    audit_imports: list[str] = []
    for relative in _AUDIT_ROUTE_PATHS:
        path = fixture.package / relative
        files.setdefault(path, b"VALUE = 1\n")
        if relative.startswith("src/") and not relative.endswith("/__init__.py"):
            audit_imports.append(
                relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
            )
    audit_graph = ("\n".join(f"import {module}" for module in audit_imports) + "\n").encode()
    files[fixture.package / "scripts/audit_camera_registration.py"] = audit_graph
    files[fixture.package / "scripts/audit_joint_equivalence_read_only.py"] = audit_graph
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _source_specs(fixture: LineageFixture) -> list[tuple[str, str, str]]:
    counts = {"package": 0, "project": 0, "runtime": 0}
    prefixes = {
        "package": "route_source",
        "project": "stanford_source",
        "runtime": "robomimic_source",
    }
    result: list[tuple[str, str, str]] = []
    for record in derive_route_closure(fixture.roots):
        index = counts[record.scope]
        counts[record.scope] += 1
        result.append((f"{prefixes[record.scope]}_{index:03d}", record.scope, record.path))
    return result


def _fixture(tmp_path: Path) -> LineageFixture:
    fixture = LineageFixture(
        tmp_path / "artifacts",
        tmp_path / "package",
        tmp_path / "project",
        tmp_path / "runtime",
        tmp_path / "package/configs/provenance/sim_to_real_400k_lineage.json",
    )
    bundle = fixture.artifact / "inference/dp_cnn_recovered_v3"
    training = fixture.artifact / "models/local_200ep/dp_cnn/seed-0/full"
    upstream = fixture.project / "05_references/external_repos/pushT-so100"
    policy = bundle / "policy.safetensors"
    checkpoint = training / "checkpoints/latest.ckpt"
    runtime_lock = fixture.package / "environments/sim-runtime.lock"
    runtime_lock_sha256 = fixture.package / "environments/sim-runtime.lock.sha256"
    project_metadata = fixture.package / "pyproject.toml"
    uv_lock = fixture.package / "uv.lock"
    workspace_status = fixture.package / "configs/workspace_status.yaml"
    environment = upstream / "environment.yml"
    for path, content in (
        (policy, b"policy"),
        (checkpoint, b"checkpoint"),
        (runtime_lock, b"runtime-lock"),
        (runtime_lock_sha256, b"runtime-lock-sha256"),
        (project_metadata, b"project-metadata"),
        (uv_lock, b"uv-lock"),
        (workspace_status, b"workspace-status"),
        (environment, b"environment"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _write_source_graph(fixture)

    upstream_members: list[dict[str, object]] = []
    for index, relative in enumerate(("src/env_gym_ee.py", "src/helper.py")):
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"upstream-{index}".encode())
        upstream_members.append({"path": relative, "kind": "fixture", "sha256": _sha256(path)})
    provenance = {
        "schema": 1,
        "source": {"name": "pushT-so100", "head": "c" * 40, "remote": "https://example.test"},
        "environment": {"path": "environment.yml", "sha256": _sha256(environment)},
        "runtime_members": upstream_members,
        "approved_patches": [],
        "excluded_untracked": [],
    }
    provenance_path = fixture.package / "configs/provenance/pusht_so100_upstream.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_bytes(_canonical(provenance))
    identity = _identity(_sha256(runtime_lock), _sha256(provenance_path))

    shape = {
        "action": {"shape": [2]},
        "obs": {
            "agent_pos": {"shape": [5], "type": "low_dim"},
            "cam_top": {"shape": [3, 96, 96], "type": "rgb"},
        },
    }
    config = {
        "horizon": 16,
        "n_action_steps": 8,
        "n_obs_steps": 2,
        "shape_meta": shape,
        "policy": {
            "_target_": identity["policy_target"],
            "horizon": 16,
            "n_action_steps": 8,
            "n_obs_steps": 2,
            "shape_meta": shape,
        },
        "task": {"env_runner": {"n_action_steps": 8, "n_obs_steps": 2}},
        "training": {"max_train_steps": 400_000},
    }
    config_path = bundle / "resolved_config.json"
    config_path.write_bytes(_canonical(config))
    bundle_manifest = {
        "schema": 1,
        "identity": identity,
        "resolved_config_sha256": _sha256(config_path),
        "source_checkpoint_sha256": _sha256(checkpoint),
    }
    (bundle / "bundle_manifest.json").write_bytes(_canonical(bundle_manifest))
    normalizer = {
        "schema": 1,
        "deployment_scope": "simulation_only",
        "identity": identity,
        "resolved_config_sha256": _sha256(config_path),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "state": {},
    }
    (bundle / "normalizer.json").write_bytes(_canonical(normalizer))
    receipt_path = training / "training_receipt.json"
    receipt_path.write_bytes(
        _canonical(
            {
                "schema": "pusht-so100-full-training-v1",
                "completed": True,
                "configured_optimizer_updates": 400_000,
                "executed_optimizer_updates": 400_000,
                "identity": identity,
                "model": "dp_cnn",
                "rollout_during_training": False,
                "training_mode": "full_production",
            }
        )
    )

    member_specs = [
        ("policy", "artifact", "inference/dp_cnn_recovered_v3/policy.safetensors"),
        ("normalizer", "artifact", "inference/dp_cnn_recovered_v3/normalizer.json"),
        ("bundle_manifest", "artifact", "inference/dp_cnn_recovered_v3/bundle_manifest.json"),
        ("resolved_config", "artifact", "inference/dp_cnn_recovered_v3/resolved_config.json"),
        (
            "source_checkpoint",
            "artifact",
            "models/local_200ep/dp_cnn/seed-0/full/checkpoints/latest.ckpt",
        ),
        (
            "training_receipt",
            "artifact",
            "models/local_200ep/dp_cnn/seed-0/full/training_receipt.json",
        ),
        ("runtime_lock", "package", "environments/sim-runtime.lock"),
        ("project_metadata", "package", "pyproject.toml"),
        ("uv_lock", "package", "uv.lock"),
        ("runtime_lock_sha256", "package", "environments/sim-runtime.lock.sha256"),
        ("workspace_status", "package", "configs/workspace_status.yaml"),
        (
            "frozen_environment_manifest",
            "package",
            "configs/provenance/pusht_so100_upstream.json",
        ),
        (
            "upstream_environment",
            "project",
            "05_references/external_repos/pushT-so100/environment.yml",
        ),
        (
            "upstream_runtime_00",
            "project",
            "05_references/external_repos/pushT-so100/src/env_gym_ee.py",
        ),
        (
            "upstream_runtime_01",
            "project",
            "05_references/external_repos/pushT-so100/src/helper.py",
        ),
        *_source_specs(fixture),
    ]
    roots = {
        "artifact": fixture.artifact,
        "package": fixture.package,
        "project": fixture.project,
        "runtime": fixture.runtime,
    }
    members = [
        {"label": label, "scope": scope, "path": path, "sha256": _sha256(roots[scope] / path)}
        for label, scope, path in member_specs
    ]
    authority_identity = {
        "artifact_id": _ARTIFACT_ID,
        "bundle_sha256": _sha256(policy),
        "camera_count": 1,
        "camera_key": "cam_top",
        "camera_resolution": [96, 96],
        "dataset_digest": identity["dataset_digest"],
        "decoded_actions": 8,
        "frozen_environment_manifest_digest": _sha256(provenance_path),
        "model": "dp_cnn",
        "observation_steps": 2,
        "optimizer_updates": 400_000,
        "policy_target": identity["policy_target"],
        "prediction_horizon": 16,
        "robomimic_commit": _ROBOMIMIC_COMMIT,
        "runtime_lock_digest": _sha256(runtime_lock),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "split_digest": identity["split_digest"],
        "stanford_commit": _STANFORD_COMMIT,
        "workspace_target": identity["workspace_target"],
    }
    index_record = {
        "bundle_path": member_specs[0][2],
        "bundle_sha256": _sha256(policy),
        "normalizer_path": member_specs[1][2],
        "normalizer_sha256": _sha256(bundle / "normalizer.json"),
        "manifest_path": member_specs[2][2],
        "manifest_sha256": _sha256(bundle / "bundle_manifest.json"),
        "config_path": member_specs[3][2],
        "config_sha256": _sha256(config_path),
        "checkpoint_path": member_specs[4][2],
        "checkpoint_sha256": _sha256(checkpoint),
        "production_receipt_path": member_specs[5][2],
        "production_receipt_sha256": _sha256(receipt_path),
        "identity": identity,
        "deployment_scope": "simulation_only",
        "training_eligible": True,
        "result_status": "anchored_final_evaluation",
    }
    index_path = fixture.artifact / "artifact-index.json"
    index_path.write_bytes(_canonical({"artifacts": {_ARTIFACT_ID: index_record}}))
    members.insert(
        0,
        {
            "label": "artifact_index",
            "scope": "artifact",
            "path": "artifact-index.json",
            "sha256": _sha256(index_path),
        },
    )
    fixture.authority.write_bytes(
        _canonical(
            {
                "schema": "so101-sim-to-real-lineage-manifest-v1",
                "artifact_id": _ARTIFACT_ID,
                "identity": authority_identity,
                "members": members,
                "runtime_fingerprint": {
                    "schema": "so101-third-party-runtime-fingerprint-v1",
                    "distributions": [],
                    "extensions": [],
                },
            }
        )
    )
    return fixture


def _validate(fixture: LineageFixture) -> ArtifactAuthorityReceipt:
    return validate_lineage(
        fixture.artifact,
        _ARTIFACT_ID,
        manifest_path=fixture.authority,
        roots=fixture.roots,
    )


def _manifest_members(fixture: LineageFixture) -> list[dict[str, object]]:
    document = read_json(fixture.authority, "test authority")
    return [
        object_mapping(member, "test member")
        for member in object_list(document["members"], "test members")
    ]


def _member_path(fixture: LineageFixture, member: dict[str, object]) -> Path:
    scope = member["scope"]
    relative = member["path"]
    assert isinstance(scope, str)
    assert isinstance(relative, str)
    roots = {
        "artifact": fixture.artifact,
        "package": fixture.package,
        "project": fixture.project,
        "runtime": fixture.runtime,
    }
    return roots[scope] / relative


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Derive mutation cases from the fixture closure instead of a copied label list."""
    if "mutation_label" not in metafunc.fixturenames:
        return
    with TemporaryDirectory(prefix="lineage-governing-surface-") as temporary:
        fixture = _fixture(Path(temporary))
        labels = [member["label"] for member in _manifest_members(fixture)]
    metafunc.parametrize("mutation_label", labels)


def test_each_declared_consumed_member_byte_drift_fails_closed(
    tmp_path: Path, mutation_label: str
) -> None:
    fixture = _fixture(tmp_path)
    member = next(item for item in _manifest_members(fixture) if item["label"] == mutation_label)
    path = _member_path(fixture, member)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(LineageError, match="digest mismatch"):
        _validate(fixture)


@pytest.mark.parametrize("label", ["normalizer", "runtime_lock"])
def test_named_core_member_tamper_regressions(tmp_path: Path, label: str) -> None:
    fixture = _fixture(tmp_path)
    member = next(item for item in _manifest_members(fixture) if item["label"] == label)
    path = _member_path(fixture, member)
    path.write_bytes(path.read_bytes() + b"named-tamper")
    with pytest.raises(LineageError, match=f"digest mismatch: {label}"):
        _validate(fixture)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("camera_count", 2),
        ("camera_resolution", [224, 224]),
        ("observation_steps", 1),
        ("prediction_horizon", 8),
        ("decoded_actions", 16),
        ("optimizer_updates", 399_999),
        ("optimizer_updates", True),
    ],
)
def test_wrong_runtime_identity_fails_closed(tmp_path: Path, field: str, bad: object) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture.authority.read_text(encoding="utf-8"))
    manifest["identity"][field] = bad
    fixture.authority.write_bytes(_canonical(manifest))
    with pytest.raises(LineageError, match="identity"):
        _validate(fixture)


def test_receipt_is_typed_deterministic_and_complete(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = _validate(fixture)
    second = _validate(fixture)
    expected_labels = {member["label"] for member in _manifest_members(fixture)}
    assert isinstance(first, ArtifactAuthorityReceipt)
    assert first.valid is True
    assert first.to_bytes() == second.to_bytes()
    assert {member.label for member in first.members} == expected_labels
    assert first.dependency_boundary == (
        "python_sources_and_loaded_extensions_exact_third_party_distributions_"
        "version_origin_bound_runtime_lock_hashed"
    )


@pytest.mark.parametrize(
    "mutation", ["omission", "duplicate-label", "duplicate-path", "self-reference"]
)
def test_manifest_member_inventory_fails_closed(tmp_path: Path, mutation: str) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture.authority.read_text(encoding="utf-8"))
    members = manifest["members"]
    if mutation == "omission":
        members.pop()
    elif mutation == "duplicate-label":
        members.append({**members[0], "path": members[1]["path"]})
    elif mutation == "duplicate-path":
        members.append({**members[0], "label": "foreign"})
    else:
        members.append(
            {
                "label": "authority_manifest",
                "scope": "package",
                "path": "configs/provenance/sim_to_real_400k_lineage.json",
                "sha256": _sha256(fixture.authority),
            }
        )
    fixture.authority.write_bytes(_canonical(manifest))
    with pytest.raises(LineageError, match=r"member|duplicate|self-reference|closure"):
        _validate(fixture)


@pytest.mark.parametrize("attack", ["escape", "absolute", "leaf-symlink", "parent-symlink"])
def test_path_escape_and_symlink_tricks_fail_closed(tmp_path: Path, attack: str) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture.authority.read_text(encoding="utf-8"))
    member = manifest["members"][1]
    if attack in {"escape", "absolute"}:
        member["path"] = "../outside" if attack == "escape" else "/etc/passwd"
        fixture.authority.write_bytes(_canonical(manifest))
    elif attack == "leaf-symlink":
        policy = _member_path(fixture, member)
        target = policy.with_name("policy-target")
        policy.rename(target)
        policy.symlink_to(target.name)
    else:
        bundle = fixture.artifact / "inference/dp_cnn_recovered_v3"
        target = fixture.artifact / "real-bundle"
        bundle.rename(target)
        bundle.symlink_to(target.name, target_is_directory=True)
    with pytest.raises(LineageError, match=r"path|symlink|escape"):
        _validate(fixture)


def test_duplicate_json_key_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw = fixture.authority.read_text(encoding="utf-8")
    fixture.authority.write_text(raw.replace('"schema":', '"schema": "duplicate",\n  "schema":', 1))
    with pytest.raises(LineageError, match="duplicate JSON key"):
        _validate(fixture)


def test_failed_validation_removes_prior_receipt_and_publishes_nothing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "authority-receipt.json"
    validate_lineage_to_file(
        fixture.artifact,
        _ARTIFACT_ID,
        output,
        manifest_path=fixture.authority,
        roots=fixture.roots,
    )
    normalizer = fixture.artifact / "inference/dp_cnn_recovered_v3/normalizer.json"
    normalizer.write_bytes(normalizer.read_bytes() + b"tamper")
    with pytest.raises(LineageError):
        validate_lineage_to_file(
            fixture.artifact,
            _ARTIFACT_ID,
            output,
            manifest_path=fixture.authority,
            roots=fixture.roots,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".*authority-receipt*.tmp"))


def test_interrupted_publication_leaves_no_accepted_partial_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "authority-receipt.json"
    original_replace = os.replace

    def interrupted_replace(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        if target == output.name:
            raise KeyboardInterrupt
        original_replace(source, target, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(os, "replace", interrupted_replace)
    with pytest.raises(KeyboardInterrupt):
        validate_lineage_to_file(
            fixture.artifact,
            _ARTIFACT_ID,
            output,
            manifest_path=fixture.authority,
            roots=fixture.roots,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".*authority-receipt*.tmp"))


def test_typed_receipt_cannot_claim_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="valid"):
        replace(_validate(_fixture(tmp_path)), valid=False)


@pytest.mark.parametrize(
    ("surface", "relative"),
    [
        ("artifact", "artifact-index.json"),
        ("package", "scripts/generate_feedback_artifacts.py"),
        (
            "project",
            (
                "05_references/external_repos/real-stanford_diffusion_policy/"
                "diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py"
            ),
        ),
    ],
)
def test_consumed_selector_or_source_drift_is_rejected(
    tmp_path: Path, surface: str, relative: str
) -> None:
    fixture = _fixture(tmp_path)
    roots = {
        "artifact": fixture.artifact,
        "package": fixture.package,
        "project": fixture.project,
    }
    consumed = roots[surface] / relative
    _validate(fixture)
    consumed.write_bytes(consumed.read_bytes() + b"drift")
    with pytest.raises(LineageError, match="digest mismatch"):
        _validate(fixture)


def test_output_parent_symlink_alias_cannot_unlink_immutable_member(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    policy = fixture.artifact / "inference/dp_cnn_recovered_v3/policy.safetensors"
    before = policy.read_bytes()
    alias = tmp_path / "output-alias"
    alias.symlink_to(policy.parent, target_is_directory=True)
    with pytest.raises(LineageError, match=r"output parent.*symlink"):
        validate_lineage_to_file(
            fixture.artifact,
            _ARTIFACT_ID,
            alias / policy.name,
            manifest_path=fixture.authority,
            roots=fixture.roots,
        )
    assert policy.is_file()
    assert policy.read_bytes() == before


def test_output_hardlink_alias_cannot_unlink_immutable_member(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    policy = fixture.artifact / "inference/dp_cnn_recovered_v3/policy.safetensors"
    alias = tmp_path / "receipt.json"
    os.link(policy, alias)
    with pytest.raises(LineageError, match="aliases an immutable"):
        validate_lineage_to_file(
            fixture.artifact,
            _ARTIFACT_ID,
            alias,
            manifest_path=fixture.authority,
            roots=fixture.roots,
        )
    assert policy.read_bytes() == b"policy"
    assert alias.is_file()


def test_existing_output_symlink_is_rejected_without_target_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    protected = fixture.artifact / "inference/dp_cnn_recovered_v3/policy.safetensors"
    output = tmp_path / "receipt.json"
    output.symlink_to(protected)
    with pytest.raises(LineageError, match="output cannot be a symlink"):
        validate_lineage_to_file(
            fixture.artifact,
            _ARTIFACT_ID,
            output,
            manifest_path=fixture.authority,
            roots=fixture.roots,
        )
    assert protected.read_bytes() == b"policy"
    assert output.is_symlink()


def test_non_directory_output_parent_is_rejected_before_unlink(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    parent = tmp_path / "not-a-directory"
    parent.write_bytes(b"preserve")
    with pytest.raises(LineageError, match="output parent"):
        validate_lineage_to_file(
            fixture.artifact,
            _ARTIFACT_ID,
            parent / "receipt.json",
            manifest_path=fixture.authority,
            roots=fixture.roots,
        )
    assert parent.read_bytes() == b"preserve"


def test_authority_manifest_symlink_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = fixture.authority.with_name("authority-target.json")
    fixture.authority.rename(target)
    fixture.authority.symlink_to(target.name)
    with pytest.raises(LineageError, match="symlink"):
        _validate(fixture)
    assert target.is_file()


def test_independently_derived_source_omission_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = read_json(fixture.authority, "test authority")
    members = object_list(manifest["members"], "test members")
    retained: list[object] = []
    removed = False
    for member in members:
        label = object_mapping(member, "test member")["label"]
        if not removed and isinstance(label, str) and label.startswith("stanford_source_"):
            removed = True
        else:
            retained.append(member)
    assert removed
    manifest["members"] = retained
    fixture.authority.write_bytes(_canonical(manifest))
    with pytest.raises(LineageError, match=r"source closure mismatch|omits a required"):
        _validate(fixture)


def test_fixture_closure_is_independently_ast_derived(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    declared = {
        (member.scope, member.path)
        for member in _validate(fixture).members
        if member.label.startswith(("route_source_", "stanford_source_", "robomimic_source_"))
    }
    assert declared == source_inventory(fixture.roots)


def test_joint_camera_audit_sources_are_in_exact_source_closure() -> None:
    closure = {path for _scope, path in source_inventory(DEFAULT_ROOTS)}
    assert set(_AUDIT_ROUTE_PATHS) <= closure


@pytest.mark.parametrize("audit_path", _AUDIT_ROUTE_PATHS)
def test_each_joint_camera_audit_source_byte_drift_rejects_before_adapter_construction(
    tmp_path: Path, audit_path: str
) -> None:
    fixture = _fixture(tmp_path)
    member = next(item for item in _manifest_members(fixture) if item["path"] == audit_path)
    path = _member_path(fixture, member)
    path.write_bytes(path.read_bytes() + b"audit-source-drift")
    adapter_constructions = 0

    def construct_after_lineage_gate() -> None:
        nonlocal adapter_constructions
        _validate(fixture)
        adapter_constructions += 1

    with pytest.raises(LineageError, match="digest mismatch"):
        construct_after_lineage_gate()
    assert adapter_constructions == 0


def test_capture_route_sources_are_in_exact_source_closure() -> None:
    closure = {path for _scope, path in source_inventory(DEFAULT_ROOTS)}
    assert set(_CAPTURE_ROUTE_PATHS) <= closure


@pytest.mark.parametrize("capture_path", _CAPTURE_ROUTE_PATHS)
def test_each_capture_route_source_byte_drift_rejects_before_adapter_construction(
    tmp_path: Path, capture_path: str
) -> None:
    fixture = _fixture(tmp_path)
    member = next(item for item in _manifest_members(fixture) if item["path"] == capture_path)
    path = _member_path(fixture, member)
    path.write_bytes(path.read_bytes() + b"capture-source-drift")
    adapter_constructions = 0

    def construct_after_lineage_gate() -> None:
        nonlocal adapter_constructions
        _validate(fixture)
        adapter_constructions += 1

    with pytest.raises(LineageError, match="digest mismatch"):
        construct_after_lineage_gate()
    assert adapter_constructions == 0


def test_runtime_preflight_contract_hashes_match_manifest() -> None:
    manifest = read_json(
        DEFAULT_ROOTS.package / "configs/provenance/sim_to_real_400k_lineage.json",
        "production authority",
    )
    members = {
        str(member["label"]): (str(member["path"]), str(member["sha256"]))
        for raw in object_list(manifest["members"], "production members")
        for member in (object_mapping(raw, "production member"),)
    }
    for label, relative in _RUNTIME_PREFLIGHT_MEMBERS.items():
        assert members[label] == (relative, _sha256(DEFAULT_ROOTS.package / relative))


@pytest.mark.parametrize("contract_path", _RUNTIME_PREFLIGHT_MEMBERS.values())
def test_runtime_contract_drift_rejects_before_device_construction(
    tmp_path: Path, contract_path: str
) -> None:
    fixture = _fixture(tmp_path)
    member = next(item for item in _manifest_members(fixture) if item["path"] == contract_path)
    path = _member_path(fixture, member)
    path.write_bytes(path.read_bytes() + b"runtime-contract-drift")
    device_constructions = 0

    def construct_after_lineage_gate() -> None:
        nonlocal device_constructions
        _validate(fixture)
        device_constructions += 1

    with pytest.raises(LineageError, match="digest mismatch"):
        construct_after_lineage_gate()
    assert device_constructions == 0


def test_process_isolated_capture_source_hashes_match_manifest() -> None:
    manifest = read_json(
        DEFAULT_ROOTS.package / "configs/provenance/sim_to_real_400k_lineage.json",
        "production authority",
    )
    members = {
        str(member["path"]): str(member["sha256"])
        for raw in object_list(manifest["members"], "production members")
        for member in (object_mapping(raw, "production member"),)
    }
    for relative in _PROCESS_CAPTURE_SOURCE_PATHS:
        source = DEFAULT_ROOTS.package / relative
        assert members[relative] == _sha256(source)


@pytest.mark.parametrize("source_path", _PROCESS_CAPTURE_SOURCE_PATHS)
def test_process_capture_source_drift_rejects_before_provider_spawn(
    tmp_path: Path, source_path: str
) -> None:
    fixture = _fixture(tmp_path)
    member = next(item for item in _manifest_members(fixture) if item["path"] == source_path)
    path = _member_path(fixture, member)
    path.write_bytes(path.read_bytes() + b"pre-spawn-source-drift")
    provider_spawns = 0

    def spawn_after_lineage_gate() -> None:
        nonlocal provider_spawns
        _validate(fixture)
        provider_spawns += 1

    with pytest.raises(LineageError, match="digest mismatch"):
        spawn_after_lineage_gate()
    assert provider_spawns == 0


def test_production_manifest_contains_governing_surface_and_derived_closure() -> None:
    package = DEFAULT_ROOTS.package
    manifest = json.loads(
        (package / "configs/provenance/sim_to_real_400k_lineage.json").read_text()
    )
    paths = {member["path"] for member in manifest["members"]}
    governing = {
        "artifact-index.json",
        "scripts/run_recovered_checkpoint_rollout.py",
        "scripts/generate_feedback_artifacts.py",
        "src/so101_pusht_benchmark/integrations/paper_baselines/runner.py",
        "src/so101_pusht_benchmark/evaluation/frozen_env.py",
        "src/so101_pusht_benchmark/core/upstream_provenance.py",
        (
            "05_references/external_repos/real-stanford_diffusion_policy/"
            "diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py"
        ),
        (
            "05_references/external_repos/real-stanford_diffusion_policy/"
            "diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py"
        ),
    }
    assert governing <= paths
    declared_sources = {
        (member["scope"], member["path"])
        for member in manifest["members"]
        if member["label"].startswith(("route_source_", "stanford_source_", "robomimic_source_"))
    }
    assert declared_sources == source_inventory(DEFAULT_ROOTS)


def test_installed_origin_outside_pinned_roots_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def escaped_spec(module: str) -> importlib.machinery.ModuleSpec:
        return importlib.machinery.ModuleSpec(
            module, loader=None, origin=str(tmp_path / "outside.py")
        )

    monkeypatch.setattr(lineage_sources.importlib.util, "find_spec", escaped_spec)
    with pytest.raises(LineageError, match="origin escapes pinned root"):
        validate_installed_origins(DEFAULT_ROOTS)


def test_implicit_package_initializers_are_in_exact_source_closure() -> None:
    declared = {path for _scope, path in source_inventory(DEFAULT_ROOTS)}
    expected = {
        "src/so101_pusht_benchmark/__init__.py",
        "src/so101_pusht_benchmark/core/__init__.py",
        "src/so101_pusht_benchmark/evaluation/__init__.py",
        "src/so101_pusht_benchmark/integrations/__init__.py",
        "src/so101_pusht_benchmark/integrations/paper_baselines/__init__.py",
        "src/so101_pusht_benchmark/sim_to_real/__init__.py",
        "src/so101_pusht_benchmark/sim_to_real/contracts.py",
        "src/so101_pusht_benchmark/training/__init__.py",
        "robomimic/models/__init__.py",
        "robomimic/utils/__init__.py",
    }
    assert expected <= declared


def test_receipt_uses_truthful_distribution_and_extension_boundary(tmp_path: Path) -> None:
    receipt = _validate(_fixture(tmp_path))
    assert receipt.dependency_boundary == (
        "python_sources_and_loaded_extensions_exact_third_party_distributions_"
        "version_origin_bound_runtime_lock_hashed"
    )
    assert hasattr(receipt, "runtime_fingerprint")


def test_nested_output_parent_swap_never_returns_misleading_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    parent = tmp_path / "nested/output"
    parent.mkdir(parents=True)
    output = parent / "receipt.json"
    renamed = parent.with_name("output-pinned")
    policy_parent = fixture.artifact / "inference/dp_cnn_recovered_v3"
    policy = policy_parent / "policy.safetensors"
    before = (_sha256(policy), policy.stat().st_dev, policy.stat().st_ino)
    original_publish = lineage_module.publish_output

    def swapped_publish(target: OutputTarget, content: bytes) -> None:
        parent.rename(renamed)
        parent.symlink_to(policy_parent, target_is_directory=True)
        original_publish(target, content)

    monkeypatch.setattr(lineage_module, "publish_output", swapped_publish)
    with pytest.raises(LineageError, match=r"output parent|published receipt"):
        validate_lineage_to_file(
            fixture.artifact,
            _ARTIFACT_ID,
            output,
            manifest_path=fixture.authority,
            roots=fixture.roots,
        )
    assert not output.exists()
    assert not (renamed / output.name).exists()
    assert (_sha256(policy), policy.stat().st_dev, policy.stat().st_ino) == before


@pytest.mark.parametrize("swap", ["symlink", "hardlink"])
def test_destination_swap_after_prepare_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swap: str
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "receipt.json"
    policy = fixture.artifact / "inference/dp_cnn_recovered_v3/policy.safetensors"
    original_publish = lineage_module.publish_output

    def swapped_publish(target: OutputTarget, content: bytes) -> None:
        if swap == "symlink":
            output.symlink_to(policy)
        else:
            os.link(policy, output)
        original_publish(target, content)

    monkeypatch.setattr(lineage_module, "publish_output", swapped_publish)
    with pytest.raises(LineageError, match=r"output|destination|alias"):
        validate_lineage_to_file(
            fixture.artifact,
            _ARTIFACT_ID,
            output,
            manifest_path=fixture.authority,
            roots=fixture.roots,
        )
    assert policy.read_bytes() == b"policy"


_TASK3_GREEN_RELATIVE = Path(".omo/start-work/evidence/task-3-green.txt")
_TASK3_GREEN_SIZE = 98
_TASK3_GREEN_SHA256 = "585b82f750b8a222dcf4af2ed72f52d2a1062430e11cebafc794d61579f9b47d"


def _task3_evidence_is_exact(package: Path) -> bool:
    evidence_root = package / ".omo"
    candidates = sorted(path for path in evidence_root.rglob("*") if "task-3" in path.name)
    if len(candidates) != 1 or candidates[0].relative_to(package) != _TASK3_GREEN_RELATIVE:
        return False
    path = candidates[0]
    try:
        stat = path.stat()
    except OSError:
        return False
    return (
        path.is_file()
        and not path.is_symlink()
        and stat.st_size == _TASK3_GREEN_SIZE
        and _sha256(path) == _TASK3_GREEN_SHA256
    )


def test_package_local_task3_evidence_requires_exact_restored_file(tmp_path: Path) -> None:
    package = tmp_path / "package"
    path = package / _TASK3_GREEN_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_bytes((DEFAULT_ROOTS.package / _TASK3_GREEN_RELATIVE).read_bytes())
    assert _task3_evidence_is_exact(package)


@pytest.mark.parametrize("mutation", ["content", "extra", "red", "symlink", "non_regular"])
def test_package_local_task3_evidence_rejects_drift_and_residue(
    tmp_path: Path, mutation: str
) -> None:
    package = tmp_path / "package"
    path = package / _TASK3_GREEN_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_bytes((DEFAULT_ROOTS.package / _TASK3_GREEN_RELATIVE).read_bytes())
    if mutation == "content":
        path.write_bytes(path.read_bytes() + b"mutated")
    elif mutation == "extra":
        (path.parent / "task-3-extra.txt").write_bytes(b"extra")
    elif mutation == "red":
        (path.parent / "task-3-red.txt").write_bytes(b"red")
    elif mutation == "symlink":
        target = path.with_name("task-3-target.txt")
        path.rename(target)
        path.symlink_to(target.name)
    else:
        path.unlink()
        path.mkdir()
    assert not _task3_evidence_is_exact(package)


@pytest.mark.parametrize(
    ("surface", "replacement"),
    [
        ("version", "9.9.9"),
        ("root", "/outside/runtime"),
        ("record_sha256", "f" * 64),
        ("extension_sha256", "e" * 64),
    ],
)
def test_runtime_fingerprint_version_origin_record_and_extension_drift_rejects(
    surface: str, replacement: str
) -> None:
    observed: dict[str, object] = {
        "schema": "so101-third-party-runtime-fingerprint-v1",
        "distributions": [
            {
                "name": "Example",
                "version": "1.0.0",
                "root": "/approved/runtime",
                "metadata_path": "/approved/runtime/example/METADATA",
                "metadata_sha256": "a" * 64,
                "record_path": "/approved/runtime/example/RECORD",
                "record_sha256": "b" * 64,
            }
        ],
        "extensions": [
            {
                "modules": ["example._native"],
                "path": "/approved/runtime/example/_native.so",
                "sha256": "c" * 64,
            }
        ],
    }
    declared = json.loads(json.dumps(observed))
    if surface == "version":
        declared["distributions"][0]["version"] = replacement
    elif surface == "root":
        declared["distributions"][0]["root"] = replacement
    elif surface == "record_sha256":
        declared["distributions"][0]["record_sha256"] = replacement
    else:
        declared["extensions"][0]["sha256"] = replacement
    with pytest.raises(LineageError, match="runtime fingerprint mismatch"):
        validate_runtime_fingerprint(declared, observed)


def test_consumed_dependency_distribution_drift_still_rejects() -> None:
    observed = observe_runtime(DEFAULT_ROOTS).fingerprint
    declared = json.loads(json.dumps(observed))
    numpy_record = next(record for record in declared["distributions"] if record["name"] == "numpy")
    numpy_record["version"] = "0.0.0-drift"
    with pytest.raises(LineageError, match="runtime fingerprint mismatch"):
        validate_runtime_fingerprint(declared, observed)


def test_fresh_subprocess_trace_equals_declared_source_closure_independently() -> None:
    modules = (
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
    traced = trace_modules(DEFAULT_ROOTS, modules)
    observed = observed_source_inventory(DEFAULT_ROOTS, traced)
    manifest = read_json(
        DEFAULT_ROOTS.package / "configs/provenance/sim_to_real_400k_lineage.json",
        "production authority",
    )
    declared = {
        (member["scope"], member["path"])
        for raw in object_list(manifest["members"], "production members")
        for member in (object_mapping(raw, "production member"),)
        if isinstance(member["scope"], str)
        and isinstance(member["path"], str)
        and isinstance(member["label"], str)
        and member["label"].startswith(("route_source_", "stanford_source_", "robomimic_source_"))
    }
    assert observed == declared


def test_renderer_backend_import_choice_does_not_change_runtime_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = (
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
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    default = runtime_fingerprint(trace_modules(DEFAULT_ROOTS, modules))
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    egl = runtime_fingerprint(trace_modules(DEFAULT_ROOTS, modules))
    assert default == egl


def test_feetech_and_serial_runtime_versions_origins_and_fingerprint_are_bound() -> None:
    runtime_root = DEFAULT_ROOTS.runtime.absolute()
    feetech = importlib.metadata.distribution("feetech-servo-sdk")
    serial = importlib.metadata.distribution("pyserial")
    assert feetech.version == "1.0.0"
    assert serial.version == "3.5"
    assert Path(str(feetech.locate_file(""))).absolute() == runtime_root
    assert Path(str(serial.locate_file(""))).absolute() == runtime_root
    owners = importlib.metadata.packages_distributions()
    assert owners.get("scservo_sdk") == ["feetech-servo-sdk"]
    spec = importlib.util.find_spec("scservo_sdk")
    assert spec is not None
    assert spec.origin == str(runtime_root / "scservo_sdk/__init__.py")

    fingerprint = observe_runtime(DEFAULT_ROOTS).fingerprint
    distributions = {
        str(record["name"]): record
        for raw in object_list(fingerprint["distributions"], "runtime distributions")
        for record in (object_mapping(raw, "runtime distribution"),)
    }
    assert distributions["feetech-servo-sdk"]["version"] == "1.0.0"
    assert distributions["pyserial"]["version"] == "3.5"


def test_fresh_trace_derives_complete_distribution_fingerprint() -> None:
    observed = observe_runtime(DEFAULT_ROOTS).fingerprint
    distributions = object_list(observed["distributions"], "observed distributions")
    names = {
        str(object_mapping(item, "observed distribution")["name"]).lower() for item in distributions
    }
    required = {
        "numpy",
        "hydra-core",
        "omegaconf",
        "diffusers",
        "einops",
        "matplotlib",
        "wandb",
        "tqdm",
        "robomimic",
    }
    assert required <= names
    manifest = read_json(
        DEFAULT_ROOTS.package / "configs/provenance/sim_to_real_400k_lineage.json",
        "production authority",
    )
    assert manifest["runtime_fingerprint"] == observed


@pytest.mark.parametrize("replacement", ["directory", "symlink"])
def test_final_lexical_parent_hook_swap_is_detected_and_published_inode_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    fixture = _fixture(tmp_path)
    parent = tmp_path / "nested/output"
    parent.mkdir(parents=True)
    pinned_parent = parent.with_name("output-pinned")
    output = parent / "receipt.json"
    policy_parent = fixture.artifact / "inference/dp_cnn_recovered_v3"
    policy = policy_parent / "policy.safetensors"
    protected_before = (_sha256(policy), policy.stat().st_dev, policy.stat().st_ino)
    original_verify = lineage_publish.lexical_parent_verifier()
    calls = 0

    def final_hook(target: OutputTarget) -> int:
        nonlocal calls
        descriptor = original_verify(target)
        calls += 1
        if calls == 3:
            parent.rename(pinned_parent)
            if replacement == "directory":
                parent.mkdir()
            else:
                parent.symlink_to(policy_parent, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(lineage_publish, "_verify_lexical_parent", final_hook)
    with pytest.raises(LineageError, match=r"output parent|published receipt"):
        validate_lineage_to_file(
            fixture.artifact,
            _ARTIFACT_ID,
            output,
            manifest_path=fixture.authority,
            roots=fixture.roots,
        )
    assert not output.exists()
    assert not (pinned_parent / output.name).exists()
    assert (_sha256(policy), policy.stat().st_dev, policy.stat().st_ino) == protected_before


def test_keyboard_interrupt_after_real_replace_removes_task_published_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "receipt.json"
    policy = fixture.artifact / "inference/dp_cnn_recovered_v3/policy.safetensors"
    protected_before = (_sha256(policy), policy.stat().st_dev, policy.stat().st_ino)
    real_replace = os.replace

    def interrupt_after_replace(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        real_replace(source, target, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        raise KeyboardInterrupt

    monkeypatch.setattr(lineage_publish.os, "replace", interrupt_after_replace)
    with pytest.raises(KeyboardInterrupt):
        validate_lineage_to_file(
            fixture.artifact,
            _ARTIFACT_ID,
            output,
            manifest_path=fixture.authority,
            roots=fixture.roots,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".*receipt*.tmp"))
    assert (_sha256(policy), policy.stat().st_dev, policy.stat().st_ino) == protected_before
