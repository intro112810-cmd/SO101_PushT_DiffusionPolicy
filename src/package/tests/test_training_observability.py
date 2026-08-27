from __future__ import annotations

import json
from pathlib import Path

from so101_pusht_benchmark.training.observability import (
    RunMetadata,
    ResourceSampler,
    write_run_metadata,
)


def test_run_metadata_is_machine_readable_and_complete(tmp_path: Path) -> None:
    output = tmp_path / "run_metadata.json"
    write_run_metadata(
        output,
        RunMetadata(
            run_id="dp-cnn-seed-0",
            model="dp_cnn",
            training_seed=0,
            dataset_digest="a" * 64,
            split_digest="b" * 64,
            runtime_digest="c" * 64,
            configured_budget={"unit": "epochs", "value": 3000},
            host="lab",
            systemd_unit="kihyun-pusht-dp-cnn-seed-0-train.service",
        ),
    )

    value = json.loads(output.read_text())
    assert value["schema"] == "pusht-training-run-metadata-v1"
    assert value["training_seed"] == 0
    assert value["configured_budget"] == {"unit": "epochs", "value": 3000}
    assert value["started_at"].endswith("Z")


def test_resource_sampler_writes_analysis_ready_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "resource_samples.jsonl"
    sampler = ResourceSampler(output, sample=lambda: {
        "gpu_utilization_percent": 91,
        "gpu_memory_used_mib": 12345,
        "gpu_power_watts": 321.5,
        "cpu_percent": 42.0,
        "rss_bytes": 987654,
        "system_memory_used_bytes": 123456789,
    })

    sampler.capture(global_step=17, epoch=3)

    value = json.loads(output.read_text())
    assert value["global_step"] == 17
    assert value["epoch"] == 3
    assert value["gpu_memory_used_mib"] == 12345
    assert value["elapsed_seconds"] >= 0
