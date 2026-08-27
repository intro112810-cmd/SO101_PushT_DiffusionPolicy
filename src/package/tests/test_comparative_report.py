from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from collections.abc import Callable, Iterator
from typing import cast

import pytest

from so101_pusht_benchmark.evaluation.comparative_report import (
    ComparisonError,
    MODEL_ORDER,
    load_comparison_inputs,
    validate_existing_comparative_report_from_index,
    write_comparative_report,
    write_comparative_report_from_index,
)
from so101_pusht_benchmark.native_cli import main as native_cli_main
from so101_pusht_benchmark.training.artifacts import (
    ArtifactError,
    ArtifactIndex,
    ArtifactScope,
    BundleFiles,
)
from so101_pusht_benchmark.training import validate_production_resume_artifact
from so101_pusht_benchmark.training import evaluator as evaluator_module
from so101_pusht_benchmark.workspace import runtime_artifact_root
from conftest import canonical_final_state_snapshot


FIXTURE_SOURCE_DIR = Path(__file__).parent / "fixtures/four_model_results"
FIXTURE_DIR = FIXTURE_SOURCE_DIR
PACKAGE_ROOT = Path(__file__).parents[1]
RUNTIME_LOCK = PACKAGE_ROOT / "environments/sim-runtime.lock"
ENVIRONMENT_MANIFEST = PACKAGE_ROOT / "configs/provenance/pusht_so100_upstream.json"


@pytest.fixture(autouse=True)
def secure_receipt_home(secure_test_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from so101_pusht_benchmark.training import artifacts

    class TestAccount:
        pw_dir = str(secure_test_home)

    def account_lookup(_uid: int) -> TestAccount:
        return TestAccount()

    monkeypatch.setattr(artifacts.pwd, "getpwuid", account_lookup)


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


@pytest.fixture(autouse=True)
def current_fixture_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "current-four-model-results"
    shutil.copytree(FIXTURE_SOURCE_DIR, destination)
    runtime_digest = hashlib.sha256(RUNTIME_LOCK.read_bytes()).hexdigest()
    environment_digest = hashlib.sha256(ENVIRONMENT_MANIFEST.read_bytes()).hexdigest()
    for path in destination.glob("*.json"):
        document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        metrics = cast("dict[str, object]", document["metrics"])
        metrics_identity = cast("dict[str, object]", metrics["identity"])
        record = cast("dict[str, object]", document["artifact_record"])
        record_identity = cast("dict[str, object]", record["identity"])
        for identity in (metrics_identity, record_identity):
            identity["runtime_lock_digest"] = runtime_digest
            identity["environment_manifest_digest"] = environment_digest
        record["metrics_sha256"] = hashlib.sha256(_canonical_bytes(metrics)).hexdigest()
        path.write_bytes(_canonical_bytes(document))
    monkeypatch.setattr(sys.modules[__name__], "FIXTURE_DIR", destination)


def _rewrite_payload_digest(document: dict[str, object]) -> None:
    metrics = document["metrics"]
    record = cast("dict[str, object]", document["artifact_record"])
    record["metrics_sha256"] = hashlib.sha256(_canonical_bytes(metrics)).hexdigest()


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "inputs"
    shutil.copytree(FIXTURE_DIR, destination)
    return destination


def _document(directory: Path, model: str) -> tuple[Path, dict[str, object]]:
    path = directory / f"{model}.json"
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return path, cast("dict[str, object]", value)


def _write_document(path: Path, document: dict[str, object]) -> None:
    path.write_bytes(_canonical_bytes(document))


@pytest.fixture
def report_destination(canonical_test_root: Path) -> Iterator[Callable[[str], Path]]:
    created: list[Path] = []

    def destination(label: str) -> Path:
        path = runtime_artifact_root() / "reports" / f"{canonical_test_root.name}-{label}"
        created.extend((path, path.with_name(f".{path.name}.tmp")))
        return path

    yield destination
    for path in reversed(created):
        shutil.rmtree(path, ignore_errors=True)


def test_artifact_index_route_consumes_real_anchored_metrics_files(
    report_destination: Callable[[str], Path],
) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        index, artifact_ids = _production_index(Path(temporary))
        output = report_destination("artifact-index")

        paths = write_comparative_report_from_index(index, artifact_ids, output)

        assert [path.name for path in paths] == ["comparison.json", "comparison.md"]
        report = json.loads(paths[0].read_text(encoding="utf-8"))
        rows = cast("list[dict[str, object]]", report["models"])
        assert [row["model"] for row in rows] == list(MODEL_ORDER)


def test_timed_evaluator_metrics_are_directly_comparator_compatible(tmp_path: Path) -> None:
    directory = _copy_fixture(tmp_path)
    path, document = _document(directory, "dp_cnn")
    metrics = cast("dict[str, object]", document["metrics"])
    metrics.pop("wall_time_s")
    times = iter((20.0, 23.25))
    timed = evaluator_module.timed_runner_result(
        lambda: metrics,
        lambda: next(times),
        lambda: None,
    )
    document["metrics"] = timed
    _rewrite_payload_digest(document)
    _write_document(path, document)

    inputs = load_comparison_inputs(directory)

    dp_cnn = next(item for item in inputs if item.model == "dp_cnn")
    assert dp_cnn.wall_time_s == 3.25


def test_four_anchored_results_emit_byte_identical_ordered_json_and_markdown(
    report_destination: Callable[[str], Path],
) -> None:
    first = report_destination("first")
    second = report_destination("second")
    first_files = write_comparative_report(FIXTURE_DIR, first)
    write_comparative_report(FIXTURE_DIR, second)

    assert [item.name for item in first_files] == ["comparison.json", "comparison.md"]
    assert (first / "comparison.json").read_bytes() == (second / "comparison.json").read_bytes()
    assert (first / "comparison.md").read_bytes() == (second / "comparison.md").read_bytes()

    report = json.loads((first / "comparison.json").read_text(encoding="utf-8"))
    rows = cast("list[dict[str, object]]", report["models"])
    assert [row["model"] for row in rows] == list(MODEL_ORDER)
    assert [row["horizons"] for row in rows] == [
        {"observation_steps": 2, "horizon": 16, "executed_actions": 8},
        {"observation_steps": 2, "horizon": 16, "executed_actions": 8},
        {"observation_steps": 2, "horizon": 2, "executed_actions": 1},
        {"observation_steps": 10, "horizon": 10, "executed_actions": 1},
    ]
    assert [row["optimizer_updates"] for row in rows] == [1_794_000, 1_794_000, 100_000, 300_000]
    assert all(
        set(row)
        == {
            "artifact_id",
            "model",
            "horizons",
            "optimizer_updates",
            "wall_time_s",
            "success_rate",
            "mean_dxy",
            "mean_dyaw",
            "mean_duration_s",
            "rollout_aggregates",
        }
        for row in rows
    )
    markdown = (first / "comparison.md").read_text(encoding="utf-8")
    assert [line.split("|")[1].strip() for line in markdown.splitlines() if line.startswith("|")][
        2:
    ] == list(MODEL_ORDER)


@pytest.mark.parametrize(
    ("label", "match"),
    [
        ("dataset", "dataset digest mismatch"),
        ("split", "split digest mismatch"),
        ("environment", "environment manifest identity mismatch"),
        ("runtime", "runtime lock identity mismatch"),
        ("seed-order", "ordered 100000..100099"),
        ("cap", "step cap"),
        ("fps", "FPS"),
        ("schema", "metric schema"),
        ("status", "anchored_final_evaluation"),
        ("fixture", "anchored_final_evaluation"),
        ("optimizer", "pinned profile"),
        ("smoke-updates", "full-production update budget"),
        ("horizon", "horizon identity mismatch"),
    ],
)
def test_mixed_or_ineligible_provenance_rejects_without_output(
    tmp_path: Path,
    report_destination: Callable[[str], Path],
    label: str,
    match: str,
) -> None:
    inputs = _copy_fixture(tmp_path)
    path, document = _document(inputs, "ibc")
    metrics = cast("dict[str, object]", document["metrics"])
    identity = cast("dict[str, object]", metrics["identity"])
    record = cast("dict[str, object]", document["artifact_record"])
    if label == "dataset":
        identity["dataset_digest"] = "c" * 64
    elif label == "split":
        identity["split_digest"] = "d" * 64
    elif label == "environment":
        identity["environment_manifest_digest"] = "e" * 64
    elif label == "runtime":
        identity["runtime_lock_digest"] = "f" * 64
    elif label == "seed-order":
        cast("list[object]", metrics["evaluation_seeds"]).reverse()
    elif label == "cap":
        metrics["step_cap"] = 299
    elif label == "fps":
        metrics["fps"] = 9
    elif label == "schema":
        metrics["metric_schema"] = "pusht-so100-dxy-dyaw-v2"
    elif label == "status":
        record["result_status"] = "production_smoke_complete_nonfinal"
    elif label == "fixture":
        record.update(
            {
                "training_eligible": False,
                "comparison_eligible": False,
                "result_status": "ineligible_fixture",
            }
        )
    elif label == "optimizer":
        identity["optimizer_updates"] = 2
    elif label == "smoke-updates":
        identity["optimizer_updates"] = 1
        metrics["optimizer_updates"] = 1
    elif label == "horizon":
        metrics["horizon"] = 15
    if label in {"dataset", "split", "environment", "runtime", "optimizer", "smoke-updates"}:
        record["identity"] = deepcopy(identity)
    _rewrite_payload_digest(document)
    _write_document(path, document)
    output = report_destination(label)

    with pytest.raises(ComparisonError, match=match):
        write_comparative_report(inputs, output)
    assert not output.exists()


def test_duplicate_missing_and_unknown_models_reject_without_output(
    tmp_path: Path, report_destination: Callable[[str], Path]
) -> None:
    for label in ("duplicate", "missing", "unknown"):
        inputs = _copy_fixture(tmp_path / label)
        if label == "missing":
            (inputs / "ibc.json").unlink()
        elif label == "duplicate":
            shutil.copyfile(inputs / "ibc.json", inputs / "duplicate.json")
        else:
            path, document = _document(inputs, "ibc")
            metrics = cast("dict[str, object]", document["metrics"])
            metrics["model"] = "unknown"
            _rewrite_payload_digest(document)
            _write_document(path, document)
        output = report_destination(label)
        with pytest.raises(ComparisonError, match="model set"):
            write_comparative_report(inputs, output)
        assert not output.exists()


def test_metrics_and_artifact_identity_tamper_reject_without_output(
    tmp_path: Path, report_destination: Callable[[str], Path]
) -> None:
    inputs = _copy_fixture(tmp_path)
    path, document = _document(inputs, "dp_cnn")
    metrics = cast("dict[str, object]", document["metrics"])
    metrics["eval/mean_dxy"] = 999.0
    _write_document(path, document)
    output = report_destination("payload-tamper")
    with pytest.raises(ComparisonError, match="metrics digest mismatch"):
        write_comparative_report(inputs, output)
    assert not output.exists()

    document = cast("dict[str, object]", json.loads((FIXTURE_DIR / "dp_cnn.json").read_text()))
    record = cast("dict[str, object]", document["artifact_record"])
    identity = cast("dict[str, object]", record["identity"])
    identity["model"] = "ibc"
    _write_document(path, document)
    with pytest.raises(ComparisonError, match="artifact and metrics identities differ"):
        write_comparative_report(inputs, output)
    assert not output.exists()


def test_arbitrary_output_stale_destination_stale_transaction_and_cancel_reject_cleanly(
    tmp_path: Path, report_destination: Callable[[str], Path]
) -> None:
    arbitrary = tmp_path / "outside"
    with pytest.raises(ComparisonError, match="reports root"):
        write_comparative_report(FIXTURE_DIR, arbitrary)
    assert not arbitrary.exists()

    stale = report_destination("stale")
    stale.mkdir()
    marker = stale / "keep"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(ComparisonError, match="already exists"):
        write_comparative_report(FIXTURE_DIR, stale)
    assert marker.read_text(encoding="utf-8") == "unchanged"

    transaction = report_destination("transaction")
    staging = transaction.with_name(f".{transaction.name}.tmp")
    staging.mkdir()
    foreign = staging / "foreign"
    foreign.write_text("unchanged", encoding="utf-8")
    with pytest.raises(ComparisonError, match="staging already exists"):
        write_comparative_report(FIXTURE_DIR, transaction)
    assert foreign.read_text(encoding="utf-8") == "unchanged"
    staging.rename(staging.with_name(f"{staging.name}-inspected"))
    shutil.rmtree(staging.with_name(f"{staging.name}-inspected"))

    cancelled = report_destination("cancelled")

    def cancel_after_json(_: Path) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        write_comparative_report(FIXTURE_DIR, cancelled, after_json_write=cancel_after_json)
    assert not cancelled.exists()
    assert not cancelled.with_name(f".{cancelled.name}.tmp").exists()
    write_comparative_report(FIXTURE_DIR, cancelled)
    assert (cancelled / "comparison.json").is_file()


def _production_index(artifact_root: Path) -> tuple[ArtifactIndex, tuple[str, ...]]:
    if artifact_root.resolve() == runtime_artifact_root().resolve():
        raise AssertionError("synthetic production fixtures cannot use the canonical artifact root")
    artifact_ids = tuple(f"{model.replace('_', '-')}-production" for model in MODEL_ORDER)
    index_path = artifact_root / "artifact-index.json"
    index_path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
    index = ArtifactIndex(index_path, artifact_root)
    for model, artifact_id in zip(MODEL_ORDER, artifact_ids, strict=True):
        document = cast(
            "dict[str, object]",
            json.loads((FIXTURE_DIR / f"{model}.json").read_text(encoding="utf-8")),
        )
        record = cast("dict[str, object]", document["artifact_record"])
        identity = cast("dict[str, object]", record["identity"])
        training = artifact_root / "models" / model / "full"
        (training / "checkpoints").mkdir(parents=True)
        checkpoint = training / "checkpoints/latest.ckpt"
        checkpoint.write_bytes(f"trusted-{model}".encode())
        config = training / "resolved_config.json"
        config.write_text("{}\n", encoding="utf-8")
        production_receipt = training / "training_receipt.json"
        production_receipt.write_text(
            json.dumps(
                {
                    "schema": "pusht-so100-full-training-v1",
                    "model": model,
                    "training_mode": "full_production",
                    "configured_optimizer_updates": {
                        "dp_cnn": 1_794_000,
                        "dp_transformer": 1_794_000,
                        "ibc": 100_000,
                        "lstm_gmm": 300_000,
                    }[model],
                    "rollout_during_training": False,
                    "completed": True,
                    "identity": identity,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        training_log = training / "logs.json.txt"
        training_log.write_text('{"loss":0.1}\n', encoding="utf-8")
        index.anchor_checkpoint(
            artifact_id,
            checkpoint,
            ArtifactScope(
                config=config,
                training_mode="full_production",
                identity=identity,
                production_receipt=production_receipt,
                training_log=training_log,
            ),
        )
        bundle_root = artifact_root / "models" / model / "bundle"
        bundle_root.mkdir(parents=True)
        bundle = bundle_root / "policy.safetensors"
        bundle.write_bytes(f"bundle-{model}".encode())
        bundle_config = bundle_root / "resolved_config.json"
        bundle_config.write_text("{}\n", encoding="utf-8")
        normalizer = bundle_root / "normalizer.json"
        normalizer.write_text("{}\n", encoding="utf-8")
        manifest = bundle_root / "bundle_manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        index.anchor_bundle(
            artifact_id,
            BundleFiles(bundle, bundle_config, normalizer, manifest),
            ArtifactScope(training_mode="full_production", identity=identity),
        )
        metrics_path = artifact_root / "evaluations" / model / "metrics.json"
        metrics_path.parent.mkdir(parents=True)
        metrics = cast("dict[str, object]", document["metrics"])
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rollouts = cast("list[dict[str, object]]", metrics["rollouts"])
        traces = metrics_path.with_name("failure_traces.json")
        traces.write_text(
            json.dumps(
                [item for item in rollouts if item["success"] is not True],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        index.anchor_evaluation(artifact_id, metrics_path, identity=identity, failure_traces=traces)
    return index, artifact_ids


def test_synthetic_production_helper_rejects_canonical_root_without_writes() -> None:
    before = canonical_final_state_snapshot()

    with pytest.raises(AssertionError, match="cannot use the canonical artifact root"):
        _production_index(runtime_artifact_root())

    assert canonical_final_state_snapshot() == before


@pytest.mark.parametrize(
    "tamper", ["metrics-bytes", "failure-traces", "status", "unexpected", "foreign"]
)
def test_evaluation_resume_tamper_fails_closed_without_writes(
    canonical_test_root: Path, tamper: str
) -> None:
    index, artifact_ids = _production_index(canonical_test_root)
    artifact_id = artifact_ids[0]
    output = canonical_test_root / "evaluations/dp_cnn"
    metrics = output / "metrics.json"
    traces = output / "failure_traces.json"
    if tamper == "metrics-bytes":
        metrics.chmod(0o600)
        metrics.write_bytes(metrics.read_bytes() + b" ")
    elif tamper == "failure-traces":
        traces.chmod(0o600)
        traces.write_bytes(b"[]\n")
    elif tamper == "status":
        raw = json.loads(index.path.read_text(encoding="utf-8"))
        raw["artifacts"][artifact_id]["result_status"] = "full_training_bundle_ready"
        index.path.write_text(json.dumps(raw), encoding="utf-8")
    elif tamper == "unexpected":
        (output / "foreign").write_bytes(b"keep")
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises((ArtifactError, ComparisonError)):
        validate_production_resume_artifact(
            index,
            stage="evaluation",
            model="ibc" if tamper == "foreign" else "dp_cnn",
            artifact_id=artifact_id,
            output=output,
        )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize(
    "tamper", ["forged-index", "receipt-bytes", "receipt-mode", "receipt-symlink", "extra-receipt"]
)
def test_evaluation_resume_requires_authenticated_final_stage_receipt(
    canonical_test_root: Path, tmp_path: Path, tamper: str
) -> None:
    index, artifact_ids = _production_index(canonical_test_root)
    artifact_id = artifact_ids[0]
    output = canonical_test_root / "evaluations/dp_cnn"
    receipt = index.stage_receipt_path(artifact_id, "evaluation")
    if tamper == "forged-index":
        raw = json.loads(index.path.read_text(encoding="utf-8"))
        raw["artifacts"][artifact_id]["comparison_eligible"] = False
        index.path.write_text(json.dumps(raw), encoding="utf-8")
    elif tamper == "receipt-bytes":
        receipt.chmod(0o600)
        receipt.write_text("{}\n", encoding="utf-8")
        receipt.chmod(0o400)
    elif tamper == "receipt-mode":
        receipt.chmod(0o600)
    elif tamper == "receipt-symlink":
        foreign = tmp_path / "foreign-evaluation-receipt.json"
        foreign.write_bytes(receipt.read_bytes())
        foreign.chmod(0o400)
        receipt.unlink()
        receipt.symlink_to(foreign)
    else:
        extra = index.receipt_root / "extra.json"
        extra.write_text("{}\n", encoding="utf-8")
        extra.chmod(0o400)
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises((ArtifactError, ComparisonError)):
        validate_production_resume_artifact(
            index,
            stage="evaluation",
            model="dp_cnn",
            artifact_id=artifact_id,
            output=output,
        )

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_existing_report_validates_for_deterministic_reuse_without_writes(
    report_destination: Callable[[str], Path],
) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        index, artifact_ids = _production_index(Path(temporary))
        output = report_destination("resume-valid")
        write_comparative_report_from_index(index, artifact_ids, output)
        before = {path.name: path.read_bytes() for path in output.iterdir()}

        paths = validate_existing_comparative_report_from_index(index, artifact_ids, output)

        assert [path.name for path in paths] == ["comparison.json", "comparison.md"]
        assert {path.name: path.read_bytes() for path in output.iterdir()} == before
        report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        assert [row["artifact_id"] for row in report["models"]] == list(artifact_ids)


@pytest.mark.parametrize(
    "tamper",
    [
        "json-bytes",
        "markdown-bytes",
        "model-identity",
        "dataset-digest",
        "split-digest",
        "environment-digest",
        "runtime-digest",
        "seeds",
        "step-cap",
        "fps",
        "metric-schema",
        "canonical-path",
        "partial",
        "unexpected-file",
        "symlinked-file",
        "foreign-report",
    ],
)
def test_existing_report_tamper_fails_closed_and_preserves_every_byte(
    report_destination: Callable[[str], Path], tmp_path: Path, tamper: str
) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        index, artifact_ids = _production_index(Path(temporary))
        output = report_destination(f"resume-{tamper}")
        write_comparative_report_from_index(index, artifact_ids, output)
        json_path = output / "comparison.json"
        markdown_path = output / "comparison.md"
        if tamper == "json-bytes":
            json_path.write_bytes(json_path.read_bytes() + b" ")
        elif tamper == "markdown-bytes":
            markdown_path.write_bytes(markdown_path.read_bytes() + b"foreign\n")
        elif tamper == "partial":
            markdown_path.unlink()
        elif tamper == "unexpected-file":
            (output / "foreign.txt").write_bytes(b"keep")
        elif tamper == "symlinked-file":
            markdown_path.unlink()
            markdown_path.symlink_to(tmp_path / "foreign.md")
        elif tamper == "foreign-report":
            other = report_destination(f"foreign-{tmp_path.name}")
            write_comparative_report(FIXTURE_DIR, other)
            json_path.write_bytes((other / "comparison.json").read_bytes())
            markdown_path.write_bytes((other / "comparison.md").read_bytes())
        else:
            report = cast("dict[str, object]", json.loads(json_path.read_text(encoding="utf-8")))
            provenance = cast("dict[str, object]", report["provenance"])
            rows = cast("list[dict[str, object]]", report["models"])
            if tamper == "model-identity":
                rows[0]["artifact_id"] = "foreign-production"
            elif tamper == "dataset-digest":
                provenance["dataset_digest"] = "c" * 64
            elif tamper == "split-digest":
                provenance["split_digest"] = "d" * 64
            elif tamper == "environment-digest":
                provenance["environment_manifest_digest"] = "e" * 64
            elif tamper == "runtime-digest":
                provenance["runtime_lock_digest"] = "f" * 64
            elif tamper == "seeds":
                provenance["evaluation_seeds"] = list(range(100001, 100101))
            elif tamper == "step-cap":
                provenance["step_cap"] = 299
            elif tamper == "fps":
                provenance["fps"] = 9
            elif tamper == "metric-schema":
                provenance["metric_schema"] = "foreign"
            elif tamper == "canonical-path":
                provenance["runtime_lock"] = "../foreign.lock"
            json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        before = {
            path.name: (path.readlink().as_posix() if path.is_symlink() else path.read_bytes())
            for path in output.iterdir()
        }

        with pytest.raises(ComparisonError, match="existing comparison report"):
            validate_existing_comparative_report_from_index(index, artifact_ids, output)

        after = {
            path.name: (path.readlink().as_posix() if path.is_symlink() else path.read_bytes())
            for path in output.iterdir()
        }
        assert after == before


def test_compare_models_subprocess_emits_only_json_on_stdout(
    report_destination: Callable[[str], Path], secure_test_home: Path
) -> None:
    artifact_root = runtime_artifact_root()
    with TemporaryDirectory(dir=artifact_root) as temporary:
        project_root = Path(temporary) / "project"
        canonical_root = project_root / "04_experiments/so101_pusht_benchmark"
        canonical_root.mkdir(parents=True)
        index, artifact_ids = _production_index(canonical_root)
        output = report_destination("subprocess-json")
        environment = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        }

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from pathlib import Path; "
                    "from so101_pusht_benchmark.training import artifacts; "
                    f"artifacts.pwd.getpwuid=lambda _uid:type('A',(),{{'pw_dir':{str(secure_test_home)!r}}})(); "
                    "from so101_pusht_benchmark import native_cli; "
                    f"native_cli.PROJECT_ROOT=Path({str(project_root)!r}); "
                    "raise SystemExit(native_cli.main(sys.argv[1:]))"
                ),
                "compare-models",
                "--artifact-index",
                str(index.path),
                *[
                    argument
                    for artifact_id in artifact_ids
                    for argument in ("--artifact-id", artifact_id)
                ],
                "--output",
                str(output),
            ],
            cwd=Path(__file__).parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        response = cast("dict[str, object]", json.loads(result.stdout))
        assert response["status"] == "generated"
        assert "ROBOMIMIC WARNING" not in result.stdout
        assert "ROBOMIMIC WARNING" in result.stderr


def test_compare_models_safe_cli_mode_reuses_valid_and_rejects_tamper(
    report_destination: Callable[[str], Path],
    capsys: pytest.CaptureFixture[str],
    canonical_test_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_pusht_benchmark import native_cli

    project_root = canonical_test_root / "project"
    artifact_root = project_root / "04_experiments/so101_pusht_benchmark"
    artifact_root.mkdir(parents=True)
    monkeypatch.setattr(native_cli, "PROJECT_ROOT", project_root)
    index, artifact_ids = _production_index(artifact_root)
    output = report_destination("resume-cli")
    write_comparative_report_from_index(index, artifact_ids, output)
    arguments = [
        "compare-models",
        "--artifact-index",
        str(index.path),
        *[argument for artifact_id in artifact_ids for argument in ("--artifact-id", artifact_id)],
        "--output",
        str(output),
        "--validate-existing",
    ]

    assert native_cli_main(arguments) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "reused"
    (output / "comparison.md").write_bytes(b"tampered\n")
    before = (output / "comparison.md").read_bytes()

    assert native_cli_main(arguments) == 1
    assert "FAIL CLOSED: existing comparison report bytes" in capsys.readouterr().out
    assert (output / "comparison.md").read_bytes() == before


def test_existing_report_directory_symlink_fails_closed(
    report_destination: Callable[[str], Path],
) -> None:
    with TemporaryDirectory(dir=runtime_artifact_root()) as temporary:
        index, artifact_ids = _production_index(Path(temporary))
        real = report_destination("resume-real")
        write_comparative_report_from_index(index, artifact_ids, real)
        linked = report_destination("resume-linked")
        linked.symlink_to(real, target_is_directory=True)

        with pytest.raises(ComparisonError, match="non-symlink directory"):
            validate_existing_comparative_report_from_index(index, artifact_ids, linked)

        assert linked.is_symlink()
        assert (real / "comparison.json").is_file()
        linked.unlink()


def test_input_loader_is_order_independent_but_output_model_order_is_locked(tmp_path: Path) -> None:
    inputs = _copy_fixture(tmp_path)
    for index, path in enumerate(sorted(inputs.glob("*.json"), reverse=True)):
        path.rename(inputs / f"{index}-{path.name}")
    loaded = load_comparison_inputs(inputs)
    assert tuple(item.model for item in loaded) == MODEL_ORDER
