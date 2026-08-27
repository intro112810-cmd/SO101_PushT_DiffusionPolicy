from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import cast

import pytest
import yaml

from so101_pusht_benchmark.integrations.paper_baselines.configs import (
    PROFILES,
    SHAPE_META,
    PolicyNamespaceError,
    policy_config,
    validate_shape_meta,
    workspace_config,
)
from so101_pusht_benchmark.training.model_smoke import (
    resolve_workspace_class,
    validate_model_identity,
    validate_profile_config,
    validate_profile_origin,
)
from so101_pusht_benchmark.workspace import runtime_artifact_root


def test_machine_profile_config_matches_runtime_authority() -> None:
    raw = yaml.safe_load(
        Path("configs/experiment/pusht_so100_model_profiles_v1.yaml").read_text(encoding="utf-8")
    )
    assert raw["schema"] == "pusht-so100-model-profiles-v1"
    assert raw["shape_meta"] == SHAPE_META
    assert tuple(raw["profiles"]) == tuple(PROFILES)
    for name, profile in PROFILES.items():
        item = raw["profiles"][name]
        assert (
            item["policy"] == f"{profile.policy_class.__module__}.{profile.policy_class.__name__}"
        )
        assert item["workspace"] == profile.workspace_target
        assert tuple(item["horizons"].values()) == (
            profile.observation_steps,
            profile.horizon,
            profile.executed_actions,
        )
    assert raw["smoke_modes"]["fixture"]["comparison_eligible"] is False
    production_smoke = raw["smoke_modes"]["production"]
    assert production_smoke["optimizer_updates"] == 1
    assert production_smoke["result_status"] == "production_smoke_complete_nonfinal"
    assert production_smoke["training_eligible"] is False
    assert production_smoke["comparison_eligible"] is False
    full = raw["training_modes"]["full_production"]
    assert full["optimizer_updates"] == 100_000
    assert full["explicit_max_updates_required"] is True
    assert full["rollout_during_training"] is False
    assert full["final_evaluation_required"] is True
    assert full["result_status"] == "full_training_complete"
    assert full["training_eligible"] is True


def test_four_profiles_lock_exact_classes_workspaces_and_horizons() -> None:
    expected = {
        "dp_cnn": (
            "DiffusionUnetHybridImagePolicy",
            "TrainDiffusionUnetHybridWorkspace",
            (2, 16, 8),
        ),
        "dp_transformer": (
            "DiffusionTransformerHybridImagePolicy",
            "TrainDiffusionTransformerHybridWorkspace",
            (2, 16, 8),
        ),
        "ibc": ("IbcDfoHybridImagePolicy", "TrainIbcDfoHybridWorkspace", (2, 2, 1)),
        "lstm_gmm": ("RobomimicImagePolicy", "TrainRobomimicImageWorkspace", (10, 10, 1)),
    }
    for name, profile in PROFILES.items():
        policy_name, workspace_name, horizons = expected[name]
        assert profile.policy_class.__name__ == policy_name
        assert profile.workspace_target.rsplit(".", 1)[1] == workspace_name
        assert (profile.observation_steps, profile.horizon, profile.executed_actions) == horizons
        config = workspace_config(name, runtime_artifact_root() / "native-paper-view", 0)
        validate_profile_config(name, config)


def test_profile_validator_rejects_unknown_horizon_action_and_local_fake() -> None:
    with pytest.raises(ValueError, match="unknown paper baseline"):
        resolve_workspace_class("unknown")
    config = workspace_config("dp_cnn", runtime_artifact_root() / "native-paper-view", 0)
    config["horizon"] = 15
    with pytest.raises(TypeError, match="horizon mismatch"):
        validate_profile_config("dp_cnn", config)
    config = workspace_config("dp_cnn", runtime_artifact_root() / "native-paper-view", 0)
    policy = cast("dict[str, object]", config["policy"])
    policy["shape_meta"] = {"obs": SHAPE_META["obs"], "action": {"shape": [3]}}
    with pytest.raises(PolicyNamespaceError, match="action"):
        validate_profile_config("dp_cnn", config)

    class LocalFakePolicy:
        pass

    with pytest.raises(TypeError, match="locked upstream class"):
        validate_model_identity("dp_cnn", LocalFakePolicy())


def test_profile_origin_substitution_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def substituted_origin(_symbol: object) -> str:
        return str(Path(__file__).resolve())

    monkeypatch.setattr(
        "so101_pusht_benchmark.training.model_smoke.inspect.getfile", substituted_origin
    )
    with pytest.raises(TypeError, match="origin is not the pinned upstream checkout"):
        validate_profile_origin("dp_cnn")


def test_shape_meta_is_exact_dual_camera_namespace_for_all_model_consumers() -> None:
    expected = {
        "obs": {
            "cam_top": {"shape": [3, 224, 224], "type": "rgb"},
            "cam_side": {"shape": [3, 224, 224], "type": "rgb"},
            "agent_pos": {"shape": [5], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }
    assert expected == SHAPE_META
    assert tuple(SHAPE_META) == ("obs", "action")
    assert tuple(SHAPE_META["obs"]) == ("cam_top", "cam_side", "agent_pos")
    for name in PROFILES:
        policy = policy_config(name)
        workspace = workspace_config(name, runtime_artifact_root() / "native-paper-view", 0)
        assert policy["shape_meta"] == expected
        assert workspace["shape_meta"] == expected
        workspace_policy = workspace["policy"]
        assert isinstance(workspace_policy, dict)
        assert workspace_policy["shape_meta"] == expected


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            {
                "obs": {
                    "cam_top": SHAPE_META["obs"]["cam_top"],
                    "agent_pos": SHAPE_META["obs"]["agent_pos"],
                },
                "action": {"shape": [2]},
            },
            "observation keys/order",
        ),
        (
            {
                "obs": {
                    "cam_side": SHAPE_META["obs"]["cam_side"],
                    "cam_top": SHAPE_META["obs"]["cam_top"],
                    "agent_pos": SHAPE_META["obs"]["agent_pos"],
                },
                "action": {"shape": [2]},
            },
            "observation keys/order",
        ),
        (
            {
                "obs": {**SHAPE_META["obs"], "unknown": {"shape": [1], "type": "low_dim"}},
                "action": {"shape": [2]},
            },
            "observation keys/order",
        ),
        (
            {
                "obs": {**SHAPE_META["obs"], "agent_pos": {"shape": [15], "type": "low_dim"}},
                "action": {"shape": [2]},
            },
            "agent_pos",
        ),
        ({"obs": SHAPE_META["obs"], "action": {"shape": [3]}}, "action"),
        ({"obs": SHAPE_META["obs"], "action": {"shape": [2.0]}}, "action"),
        (
            {
                "obs": {**SHAPE_META["obs"], "cam_top": {"shape": [3, 96, 96], "type": "rgb"}},
                "action": {"shape": [2]},
            },
            "cam_top",
        ),
        (
            {"obs": SHAPE_META["obs"], "action": {"shape": [2]}, "extra": {}},
            "shape_meta keys/order",
        ),
    ],
)
def test_shape_meta_rejects_single_camera_order_shapes_and_unknown_keys(
    mutation: dict[str, object], match: str
) -> None:
    with pytest.raises(PolicyNamespaceError, match=match):
        validate_shape_meta(mutation)


@pytest.mark.parametrize("name", tuple(PROFILES))
def test_shape_meta_dynamically_builds_real_upstream_dual_camera_encoder(name: str) -> None:
    project_root = Path(__file__).resolve().parents[3]
    paper_python = (
        project_root / "04_experiments/so101_pusht_benchmark/cache/envs/paper-baselines/bin/python"
    )
    driver = """
import inspect, json
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf
from so101_pusht_benchmark.integrations.paper_baselines.configs import observation_encoder, policy_config
name = __import__('sys').argv[1]
config = policy_config(name)
if name == 'dp_cnn': config.update(diffusion_step_embed_dim=32, down_dims=[32, 64])
elif name == 'dp_transformer': config.update(n_layer=1, n_head=4, n_emb=32, n_cond_layers=0)
elif name == 'ibc': config.update(train_n_neg=2, pred_n_iter=1, pred_n_samples=2)
policy = instantiate(OmegaConf.create(config))
encoder = observation_encoder(policy)
print('ORIGIN_JSON=' + json.dumps({
    'policy_module': type(policy).__module__,
    'policy_source': str(Path(inspect.getfile(type(policy))).resolve()),
    'encoder_module': type(encoder).__module__,
    'encoder_source': str(Path(inspect.getfile(type(encoder))).resolve()),
    'keys': list(encoder.obs_nets),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "03_code/so101_pusht_benchmark/src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [str(paper_python), "-c", driver, name],
        cwd=project_root / "03_code/so101_pusht_benchmark",
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(item for item in result.stdout.splitlines() if item.startswith("ORIGIN_JSON="))
    origin = json.loads(line.removeprefix("ORIGIN_JSON="))
    assert origin["keys"] == ["cam_top", "cam_side", "agent_pos"]
    assert origin["policy_module"].startswith("diffusion_policy.policy.")
    assert origin["encoder_module"] == "robomimic.models.obs_nets"
    assert "/cache/upstream/stanford/diffusion_policy/policy/" in origin["policy_source"]
    assert "/cache/envs/paper-baselines/" in origin["encoder_source"]
    assert "so101_pusht_benchmark" not in origin["encoder_module"]


def test_no_project_custom_vision_encoder_module() -> None:
    source = Path("src/so101_pusht_benchmark")
    candidates = [
        path
        for path in source.rglob("*.py")
        if "vision" in path.name.lower() or "encoder" in path.name.lower()
    ]
    assert candidates == []
