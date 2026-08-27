"""Real-shadow CLI requires authentic two-sample history before inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import cast

BENCHMARK = Path(__file__).resolve().parents[1]
FIXTURES = BENCHMARK / "tests/fixtures/sim_to_real"
SCRIPT = BENCHMARK / "scripts/run_real_shadow_inference.py"
LINEAGE = FIXTURES / "lineage.json"
JOINT = FIXTURES / "joint-equivalence.json"
CAMERA = FIXTURES / "camera-registration.json"
FRAME = FIXTURES / "physical_frame.png"
ARTIFACT = "fixture-local-dp_cnn-recovered-v3-seed0"


def _run(
    samples: Path,
    output: Path,
    *,
    lineage: Path = LINEAGE,
    artifact: str = ARTIFACT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--artifact-root",
            str(BENCHMARK),
            "--model",
            "dp_cnn",
            "--artifact",
            artifact,
            "--frame",
            str(FRAME),
            "--samples",
            str(samples),
            "--lineage",
            str(lineage),
            "--joint",
            str(JOINT),
            "--camera",
            str(CAMERA),
            "--output",
            str(output),
        ],
        cwd=BENCHMARK,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(BENCHMARK / "src")},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return cast("dict[str, object]", raw)


def test_real_shadow_cli_uses_two_distinct_sequential_samples(tmp_path: Path) -> None:
    output = tmp_path / "shadow.json"
    result = _run(FIXTURES / "synchronized_samples.json", output)

    assert result.returncode == 0, result.stderr
    receipt = _json(output)
    assert receipt["sample_ids"] == ["sample-000", "sample-001"]
    sample_digests = cast("list[str]", receipt["sample_digests"])
    assert len(set(sample_digests)) == 2
    assert len(cast("list[list[float]]", receipt["predicted_actions"])) == 8
    assert receipt["actuation_performed"] is False
    assert receipt["evidence_scope"] == "test_fixture_only"
    assert receipt["policy_evidence"] == "fixture_adapter_not_frozen_production"


def test_real_frozen_policy_route_fails_before_policy_without_bound_camera_source(
    tmp_path: Path,
) -> None:
    fixture_lineage = _json(LINEAGE)
    document = {
        "artifact_id": "local-dp_cnn-recovered-v3-seed0",
        "authority_digest": fixture_lineage["authority_digest"],
        "valid": True,
    }
    encoded = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    document["lineage_digest"] = hashlib.sha256(encoded).hexdigest()
    lineage = tmp_path / "production-lineage.json"
    lineage.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "production-shadow.json"

    result = _run(
        FIXTURES / "synchronized_samples.json",
        output,
        lineage=lineage,
        artifact="local-dp_cnn-recovered-v3-seed0",
    )

    assert result.returncode == 2
    assert "HISTORY_INCOMPLETE: camera source is not fixture-bound" in result.stderr
    assert not output.exists()


def test_real_shadow_cli_rejects_one_sample_before_policy(tmp_path: Path) -> None:
    source = _json(FIXTURES / "synchronized_samples.json")
    source["samples"] = cast("list[object]", source["samples"])[:1]
    samples = tmp_path / "one.json"
    samples.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "shadow.json"

    result = _run(samples, output)

    assert result.returncode == 2
    assert "HISTORY_INCOMPLETE" in result.stderr
    assert not output.exists()


def test_real_shadow_cli_rejects_duplicate_sample_before_policy(tmp_path: Path) -> None:
    output = tmp_path / "shadow.json"
    result = _run(FIXTURES / "duplicated_history.json", output)

    assert result.returncode == 2
    assert "R_DUPLICATE" in result.stderr
    assert not output.exists()
