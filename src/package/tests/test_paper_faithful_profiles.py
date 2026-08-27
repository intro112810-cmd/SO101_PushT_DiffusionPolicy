from __future__ import annotations

from pathlib import Path

from so101_pusht_benchmark.training.paper_profiles import load_paper_profiles


CONFIG = (
    Path(__file__).parents[1]
    / "configs/experiment/pusht_so100_paper_faithful_200ep_v1.yaml"
)


def test_paper_faithful_profiles_pin_approved_budgets_and_seeds() -> None:
    profiles = load_paper_profiles(CONFIG)

    assert profiles.training_seeds == (0, 1, 2)
    assert profiles.models["dp_cnn"].budget == {"unit": "epochs", "value": 3000}
    assert profiles.models["dp_transformer"].budget == {
        "unit": "epochs",
        "value": 3000,
    }
    assert profiles.models["ibc"].budget == {"unit": "updates", "value": 100000}
    assert profiles.models["lstm_gmm"].budget == {"unit": "updates", "value": 300000}
    assert profiles.models["dp_cnn"].resolved_optimizer_updates == 1_794_000
    assert profiles.models["dp_transformer"].resolved_optimizer_updates == 1_794_000
    assert profiles.models["ibc"].resolved_optimizer_updates == 100_000
    assert profiles.models["lstm_gmm"].resolved_optimizer_updates == 300_000


def test_paper_faithful_profiles_pin_model_specific_recipes() -> None:
    models = load_paper_profiles(CONFIG).models

    assert models["dp_cnn"].parameters["horizons"] == [2, 8, 16]
    assert models["dp_cnn"].parameters["batch_size"] == 64
    assert models["dp_transformer"].parameters["attention_dropout"] == 0.01
    assert models["dp_transformer"].parameters["weight_decay"] == 0.1
    assert models["ibc"].parameters["training_negatives"] == 256
    assert models["ibc"].parameters["inference_samples"] == 4096
    assert models["lstm_gmm"].parameters["batch_size"] == 16
    assert models["lstm_gmm"].parameters["gmm_modes"] == 5
