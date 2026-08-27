from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from so101_pusht_benchmark.native_cli import command_parser
from so101_pusht_benchmark.workspace import (
    PACKAGE_ROOT,
    PROJECT_ROOT,
    WorkspacePolicyError,
    authorize_real_diagnostic_route,
    load_workspace_policy,
)


PLAN = ".omo/plans/pusht-so100-four-model-clean-restart.md"
NATIVE_SCHEMA = "pusht-so100-native-v1"
HISTORICAL_CONFIGS = {
    "configs/benchmark/pusht_v1.yaml",
    "configs/benchmark/pusht_mouse_topdown_v3.yaml",
    "configs/collection/pusht_gamepad_v1.yaml",
    "configs/collection/pusht_mouse_keyboard_v3.yaml",
    "configs/export/paper_view_v1.yaml",
}


def test_baseline_four_model_authority_is_simulation_only() -> None:
    benchmark = yaml.safe_load(
        (PACKAGE_ROOT / "configs/benchmark/pusht_so100_native_v1.yaml").read_text(encoding="utf-8")
    )

    assert benchmark["deployment_scope"] == "simulation_only"
    assert benchmark["policy_allowlist"] == [
        "cam_top",
        "cam_side",
        "agent_pos",
        "action",
    ]


def test_clean_restart_is_the_only_governing_plan_and_contract() -> None:
    policy = load_workspace_policy()

    assert policy["schema"] == 2
    assert policy["workspace"]["plan"] == PLAN
    assert "base_plan" not in policy["workspace"]
    assert policy["native_contract"] == {
        "schema": NATIVE_SCHEMA,
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
    assert policy["runtime"]["native_lock"] == "environments/sim-runtime.lock"
    assert policy["runtime"]["fallback"] == "forbidden"


def test_root_governance_names_only_clean_restart_as_active() -> None:
    root_agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    root_readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    combined = root_agents + root_readme

    assert combined.count(PLAN) == 2
    assert (
        "approved governing plan is\n`.omo/plans/so101-pusht-z-state-realignment.md`"
        not in combined
    )
    assert "collect-sim" not in combined
    assert "historical/reference" in combined


def test_active_and_historical_config_routes_are_disjoint() -> None:
    policy = load_workspace_policy()
    active = set(policy["active_configs"].values())
    historical = set(policy["historical"]["configs"])

    assert not active & historical
    assert historical >= HISTORICAL_CONFIGS
    for relative in active:
        value = yaml.safe_load((PACKAGE_ROOT / relative).read_text(encoding="utf-8"))
        assert value["status"] == "active"
        assert value["contract_schema"] == NATIVE_SCHEMA


def test_cli_defaults_and_active_docs_do_not_route_to_historical_pipeline() -> None:
    parser = command_parser()
    validate = parser.parse_args(["validate-contract"])
    collect = parser.parse_args(["collect-native", "--preflight"])
    export = parser.parse_args(["export-native", "--preflight"])

    assert validate.config.name == "pusht_so100_native_v1.yaml"
    assert collect.config.name == "pusht_so100_native_v1.yaml"
    assert collect.collection_config.name == "pusht_so100_f710_native_v1.yaml"
    assert export.config.name == "pusht_so100_native_v1.yaml"

    active_docs = [
        PACKAGE_ROOT / "AGENTS.md",
        PACKAGE_ROOT / "README.md",
        PACKAGE_ROOT / "bridge_contract.md",
        PACKAGE_ROOT / "docs/collecting_human_pilot.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in active_docs)
    assert "configs/benchmark/pusht_mouse_topdown_v3.yaml" not in text
    assert "configs/collection/pusht_mouse_keyboard_v3.yaml" not in text
    assert "collect-sim" not in text


def test_pyproject_console_scripts_expose_only_native_runtime_routes() -> None:
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    for command in ("validate-contract", "inspect-env", "collect-native", "export-native"):
        assert f'{command} = "so101_pusht_benchmark.native_cli:main"' in pyproject
    for historical in ("validate-sim", "step-smoke", "calibrate-sim", "collect-sim"):
        assert f"{historical} =" not in pyproject


def test_todo1_cli_hunks_survive_todo2_receipt_and_todo3_lazy_dispatch() -> None:
    receipt = json.loads(
        (PACKAGE_ROOT / "configs/provenance/dirty_work_receipt.json").read_text(encoding="utf-8")
    )
    extension = receipt["todo2_intentional_cli_extension"]
    cli = PACKAGE_ROOT / "src/so101_pusht_benchmark/cli.py"
    patch = PACKAGE_ROOT / "configs/provenance/todo2_cli_extension.patch"
    cli_text = cli.read_text(encoding="utf-8")

    assert extension["current_sha256"] == (
        "04b2055db8e6a283ac88a33a274171cde98ff5ed8fa795be039f4dfecd9132f4"
    )
    assert extension["current_byte_count"] == 22398
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == extension["extension_sha256"]
    assert 'choices=("dp_cnn", "dp_transformer", "ibc", "lstm_gmm")' in cli_text
    assert 'train.add_argument("--smoke", action="store_true"' in cli_text
    assert "model=args.model" in cli_text
    assert "smoke=args.smoke" in cli_text


def test_real_diagnostic_route_is_narrow() -> None:
    policy = load_workspace_policy()
    route = authorize_real_diagnostic_route(
        policy,
        authority="separate_control_plane",
        entry_point="so101_pusht_benchmark.sim_to_real.supervisor",
    )

    assert route["deployment_scope"] == "physical_diagnostic_only"
    assert route["module_root"] == "so101_pusht_benchmark.sim_to_real"
    assert route["allowed_scripts"] == [
        "scripts/check_guarded_single_step.py",
        "scripts/run_guarded_single_step.py",
        "scripts/run_guarded_bounded_rollout.py",
        "scripts/verify_guarded_rollout.py",
    ]
    assert route["require_owner_approved_policy"] is True
    assert route["require_single_owner_writer"] is True
    with pytest.raises(WorkspacePolicyError, match="outside guarded sim-to-real route"):
        authorize_real_diagnostic_route(
            policy,
            authority="separate_control_plane",
            entry_point="so101_pusht_benchmark.training.launcher",
        )


def test_deployable_training_identity_mutation_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load((PACKAGE_ROOT / "configs/workspace_status.yaml").read_text())
    raw["model_authority"]["identities"]["dp_cnn"]["training"] = "hardware_deployable"
    mutated = tmp_path / "workspace.yaml"
    mutated.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(
        WorkspacePolicyError,
        match="simulation identity cannot control hardware",
    ) as rejection:
        load_workspace_policy(mutated)
    assert str(rejection.value) == "simulation identity cannot control hardware"


def test_malformed_diagnostic_governance_rejects(tmp_path: Path) -> None:
    raw = yaml.safe_load((PACKAGE_ROOT / "configs/workspace_status.yaml").read_text())
    del raw["real_diagnostic_rollout"]["require_owner_approved_policy"]
    malformed = tmp_path / "workspace.yaml"
    malformed.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(WorkspacePolicyError, match="diagnostic rollout governance"):
        load_workspace_policy(malformed)


def test_policy_rejects_stale_second_active_plan(tmp_path: Path) -> None:
    raw = yaml.safe_load((PACKAGE_ROOT / "configs/workspace_status.yaml").read_text())
    raw["workspace"]["base_plan"] = ".omo/plans/so101-pusht-z-state-realignment.md"
    malformed = tmp_path / "workspace.yaml"
    malformed.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(WorkspacePolicyError, match="sole governing plan"):
        load_workspace_policy(malformed)
