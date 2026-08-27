from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

import pytest

from so101_pusht_benchmark.data.splits import ExperimentConfig, freeze_training_view
from so101_pusht_benchmark.training.launcher import MODEL_NAMES
from test_model_smoke import ineligible_fixture


def _paper_python() -> Path:
    project = Path(__file__).resolve().parents[3]
    return project / "04_experiments/so101_pusht_benchmark/cache/envs/paper-baselines/bin/python"


def _store(root: Path, mode: str) -> Path:
    source = ineligible_fixture(root / "source", episodes=3 if mode == "production" else 1)
    if mode == "fixture":
        return source
    frozen, _ = freeze_training_view(
        source,
        root / "frozen",
        ExperimentConfig(
            "pusht-so100-experiment-v1",
            3,
            {
                "train": Decimal("0.34"),
                "validation": Decimal("0.33"),
                "test": Decimal("0.33"),
            },
        ),
    )
    return frozen


def _launch(root: Path, model: str, mode: str) -> subprocess.CompletedProcess[str]:
    store = _store(root, mode)
    artifact_root = root / "artifacts"
    artifact_root.mkdir()
    index = root / "index.json"
    index.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
    driver = """
import json, sys
from pathlib import Path
from so101_pusht_benchmark.training.artifacts import ArtifactIndex
from so101_pusht_benchmark.training.launcher import TrainingLaunch, launch_training
root = Path(sys.argv[1]); store = Path(sys.argv[2]); output = Path(sys.argv[3]); index_path = Path(sys.argv[4])
model = sys.argv[5]; mode = sys.argv[6]
idx = ArtifactIndex(index_path, root)
checkpoint = launch_training(store, output, idx, TrainingLaunch(0, model + '-' + mode, model=model, smoke_mode=mode))
print('LAUNCH_JSON=' + json.dumps({'checkpoint': str(checkpoint), 'record': idx.record(model + '-' + mode)}, sort_keys=True))
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "WANDB_MODE": "offline",
    }
    return subprocess.run(
        [
            str(_paper_python()),
            "-c",
            driver,
            str(artifact_root),
            str(store),
            str(artifact_root / "final"),
            str(index),
            model,
            mode,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("model", MODEL_NAMES)
@pytest.mark.parametrize("mode", ["fixture", "production"])
def test_real_launch_training_workspace_smoke(
    canonical_test_root: Path, model: str, mode: str
) -> None:
    root = canonical_test_root
    try:
        result = _launch(root, model, mode)
        assert result.returncode == 0, result.stderr
        payload = json.loads(
            next(
                line for line in result.stdout.splitlines() if line.startswith("LAUNCH_JSON=")
            ).split("=", 1)[1]
        )
        checkpoint = Path(payload["checkpoint"])
        assert checkpoint.is_file()
        assert checkpoint.parent.parent == root / "artifacts/final"
        record = payload["record"]
        assert record["comparison_eligible"] is False
        assert record["result_status"] == (
            "ineligible_fixture" if mode == "fixture" else "production_smoke_complete_nonfinal"
        )
        assert record["training_eligible"] is False
        receipt = json.loads((root / "artifacts/final/smoke_receipt.json").read_text())
        assert receipt["optimizer_steps"] == 1
        assert receipt["loss"] >= 0
        assert not any((root / "artifacts").glob(".final.tmp-*"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_real_cli_production_selection_and_bad_fixture_fail_closed(
    canonical_test_root: Path,
) -> None:
    root = canonical_test_root
    try:
        source = ineligible_fixture(root / "source", episodes=3)
        frozen, _ = freeze_training_view(
            source,
            root / "frozen",
            ExperimentConfig(
                "pusht-so100-experiment-v1",
                3,
                {
                    "train": Decimal("0.34"),
                    "validation": Decimal("0.33"),
                    "test": Decimal("0.33"),
                },
            ),
        )
        index_path = root / "index.json"
        index_path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
        environment = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WANDB_MODE": "offline",
        }

        def run_cli(
            store: Path, output: Path, artifact_id: str
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [
                    str(_paper_python()),
                    "-m",
                    "so101_pusht_benchmark.cli",
                    "train-model",
                    "--model",
                    "dp_cnn",
                    "--paper-view",
                    str(store),
                    "--output",
                    str(output),
                    "--artifact-id",
                    artifact_id,
                    "--artifact-index",
                    str(index_path),
                    "--smoke-mode",
                    "production",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

        good_output = root / "good"
        good = run_cli(frozen, good_output, "good-production")
        assert good.returncode == 0, f"{good.stdout}\n{good.stderr}"
        payload = json.loads(good.stdout.splitlines()[-1])
        assert payload["smoke_mode"] == "production"
        assert payload["artifact"]["result_status"] == "production_smoke_complete_nonfinal"
        assert payload["artifact"]["training_eligible"] is False
        assert payload["artifact"]["comparison_eligible"] is False

        bad_output = root / "bad"
        bad = run_cli(source, bad_output, "bad-production")
        assert bad.returncode == 1
        assert "FAIL CLOSED: production smoke requires immutable frozen manifest" in bad.stdout
        assert not bad_output.exists()
        assert "bad-production" not in json.loads(index_path.read_text())["artifacts"]

        tampered = root / "tampered"
        shutil.copytree(frozen, tampered)
        split_path = tampered / "splits.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["digest"] = "0" * 64
        split_path.write_text(json.dumps(split), encoding="utf-8")
        tampered_output = root / "tampered-output"
        tampered_result = run_cli(tampered, tampered_output, "tampered-production")
        assert tampered_result.returncode == 1
        assert "FAIL CLOSED:" in tampered_result.stdout
        assert "digest" in tampered_result.stdout
        assert not tampered_output.exists()
        assert "tampered-production" not in json.loads(index_path.read_text())["artifacts"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_full_production_launch_anchors_only_completed_locked_budget(
    tmp_path: Path, secure_test_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wandb

    from so101_pusht_benchmark.training import artifacts, launcher
    from so101_pusht_benchmark.training.artifacts import ArtifactIndex, require_production_artifact
    from so101_pusht_benchmark.training.model_smoke import SmokeStoreIdentity

    class TestAccount:
        pw_dir = str(secure_test_home)

    def account_lookup(_uid: int) -> TestAccount:
        return TestAccount()

    monkeypatch.setattr(artifacts.pwd, "getpwuid", account_lookup)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    index_path = artifact_root / "artifact-index.json"
    index_path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
    index = ArtifactIndex(index_path, artifact_root)
    calls: list[str] = []

    class FakeWorkspace:
        def __init__(self, _config: object, *, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self._saving_thread = None
            self.global_step = 0

        def run(self) -> None:
            calls.append("run")
            self.global_step = 1_794_000
            (self.output_dir / "logs.json.txt").write_text('{"loss":0.1}\n', encoding="utf-8")

        def save_checkpoint(self, *, use_thread: bool) -> Path:
            assert use_thread is False
            checkpoint = self.output_dir / "checkpoints/latest.ckpt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"full-production")
            return checkpoint

    config: dict[str, object] = {
        "_target_": f"{FakeWorkspace.__module__}.{FakeWorkspace.__name__}",
        "dataloader": {"batch_size": 32},
        "task": {
            "dataset": {},
            "env_runner": {"options": {}},
        },
        "training": {"max_train_steps": 1_794_000, "num_epochs": 1},
    }

    def no_runtime() -> None:
        return None

    def resolved(_model: str, _path: Path, _seed: int) -> dict[str, object]:
        return config

    def no_profile(_model: str, _value: object) -> None:
        return None

    def no_origin(_model: str) -> None:
        return None

    def workspace_class(_model: str) -> type[FakeWorkspace]:
        return FakeWorkspace

    def store_identity(_path: Path, *, mode: str) -> SmokeStoreIdentity:
        assert mode == "production"
        return SmokeStoreIdentity("production", "a" * 64, "b" * 64, True, False)

    monkeypatch.setattr(launcher, "assert_paper_runtime", no_runtime)
    monkeypatch.setattr(launcher, "resolved_config", resolved)
    monkeypatch.setattr(launcher, "validate_profile_config", no_profile)
    monkeypatch.setattr(launcher, "validate_profile_origin", no_origin)
    monkeypatch.setattr(launcher, "resolve_workspace_class", workspace_class)
    monkeypatch.setattr(launcher, "validate_smoke_store", store_identity)
    monkeypatch.setattr(wandb, "finish", no_runtime)

    output = artifact_root / "models/dp_cnn/full"
    checkpoint = launcher.launch_training(
        artifact_root / "frozen",
        output,
        index,
        launcher.TrainingLaunch(
            0,
            "dp-cnn-production",
            model="dp_cnn",
            training_mode="full_production",
            max_updates=1_794_000,
        ),
    )

    assert calls == ["run"]
    assert checkpoint == output / "checkpoints/latest.ckpt"
    receipt = json.loads((output / "training_receipt.json").read_text(encoding="utf-8"))
    assert receipt["configured_optimizer_updates"] == 1_794_000
    assert receipt["executed_optimizer_updates"] == 1_794_000
    assert receipt["rollout_during_training"] is False
    assert receipt["completed"] is True
    record = index.record("dp-cnn-production")
    assert record["result_status"] == "full_training_complete"
    record_identity = record["identity"]
    assert isinstance(record_identity, dict)
    assert record_identity["optimizer_updates"] == 1_794_000
    require_production_artifact(record, operation="bundle export")


def test_launcher_joins_overwritten_checkpoint_writer_before_immutable_rejection(
    canonical_test_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wandb

    from so101_pusht_benchmark.training import launcher
    from so101_pusht_benchmark.training.artifacts import ArtifactError, ArtifactIndex
    from so101_pusht_benchmark.training.model_smoke import SmokeStoreIdentity

    root = canonical_test_root
    artifact_root = root / "artifacts"
    artifact_root.mkdir()
    foreign = artifact_root / "foreign.marker"
    foreign.write_bytes(b"foreign")
    index_path = root / "index.json"
    index_path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
    release = threading.Event()
    writer_started = threading.Event()
    writers: list[threading.Thread] = []

    class JoinReleasesWriter(threading.Thread):
        def join(self, timeout: float | None = None) -> None:
            release.set()
            super().join(timeout)

    class FakeWorkspace:
        def __init__(self, _config: object, *, output_dir: str) -> None:
            self.output_dir = Path(output_dir)
            self._saving_thread: threading.Thread | None = None
            self._save_count = 0

        def run(self) -> None:
            self.save_checkpoint(use_thread=True)
            self.save_checkpoint(use_thread=True)

        def save_checkpoint(self, *, use_thread: bool) -> Path:
            self._save_count += 1
            if not use_thread:
                checkpoint = self.output_dir / "checkpoints/latest.ckpt"
                checkpoint.parent.mkdir(exist_ok=True)
                checkpoint.write_bytes(b"final")
                return checkpoint
            if self._save_count == 1:
                owned = (self.output_dir / "owned-writer.log").open("wb")

                def delayed_write() -> None:
                    writer_started.set()
                    release.wait()
                    owned.write(b"complete")
                    owned.close()

                writer: threading.Thread = JoinReleasesWriter(target=delayed_write)
            else:
                writer = threading.Thread(target=lambda: None)
            writer.start()
            writers.append(writer)
            self._saving_thread = writer
            return self.output_dir / f"checkpoint-{self._save_count}.ckpt"

    config: dict[str, object] = {
        "_target_": f"{FakeWorkspace.__module__}.{FakeWorkspace.__name__}",
        "training": {"max_train_steps": 1000, "num_epochs": 100},
    }

    def no_runtime() -> None:
        return None

    def resolved(_model: str, _path: Path, _seed: int) -> dict[str, object]:
        return config

    def no_profile(_model: str, _value: object) -> None:
        return None

    def no_origin(_model: str) -> None:
        return None

    def workspace_class(_model: str) -> type[FakeWorkspace]:
        return FakeWorkspace

    def unchanged_smoke(value: dict[str, object], *, mode: object) -> dict[str, object]:
        assert mode == "fixture"
        return value

    def store_identity(_path: Path, *, mode: str) -> SmokeStoreIdentity:
        assert mode == "fixture"
        return SmokeStoreIdentity("fixture", "a" * 64, None, False, False)

    monkeypatch.setattr(launcher, "assert_paper_runtime", no_runtime)
    monkeypatch.setattr(launcher, "resolved_config", resolved)
    monkeypatch.setattr(launcher, "validate_profile_config", no_profile)
    monkeypatch.setattr(launcher, "validate_profile_origin", no_origin)
    monkeypatch.setattr(launcher, "resolve_workspace_class", workspace_class)
    monkeypatch.setattr(launcher, "smoke_probe_config", unchanged_smoke)
    monkeypatch.setattr(launcher, "validate_smoke_store", store_identity)

    def run_batch(workspace: object, *_args: object, **_kwargs: object) -> dict[str, object]:
        assert isinstance(workspace, FakeWorkspace)
        workspace.run()
        return {"canonical_digest": "a" * 64, "split_digest": None}

    monkeypatch.setattr(launcher, "run_workspace_one_batch", run_batch)
    monkeypatch.setattr(wandb, "finish", lambda: None)
    output = artifact_root / "final"
    index = ArtifactIndex(index_path, artifact_root)
    checkpoint = launcher.launch_training(
        root / "store",
        output,
        index,
        launcher.TrainingLaunch(0, "first", model="dp_cnn", smoke_mode="fixture"),
    )
    assert writer_started.is_set()
    assert checkpoint == output / "checkpoints/latest.ckpt"

    def digest_tree() -> list[tuple[str, str]]:
        return [
            (str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]

    before = digest_tree()
    with pytest.raises(ArtifactError, match="output already exists"):
        launcher.launch_training(
            root / "store",
            output,
            index,
            launcher.TrainingLaunch(0, "second", model="dp_cnn", smoke_mode="fixture"),
        )
    for writer in writers:
        writer.join()
    assert digest_tree() == before
    assert foreign.read_bytes() == b"foreign"
    assert not any(writer.is_alive() for writer in writers)
    assert not list(artifact_root.glob(".final.tmp-*"))


def test_full_production_preflight_is_distinct_bounded_and_artifact_free(
    canonical_test_root: Path,
) -> None:
    root = canonical_test_root
    try:
        frozen = _store(root, "production")
        index_path = root / "index.json"
        index_path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
        output = root / "must-not-exist"
        environment = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WANDB_MODE": "offline",
        }
        base = [
            str(_paper_python()),
            "-m",
            "so101_pusht_benchmark.cli",
            "train-model",
            "--model",
            "dp_cnn",
            "--paper-view",
            str(frozen),
            "--output",
            str(output),
            "--artifact-id",
            "dp-cnn-production",
            "--artifact-index",
            str(index_path),
            "--paper-profiles",
            str(
                Path(__file__).resolve().parents[1]
                / "configs/experiment/pusht_so100_paper_faithful_200ep_v1.yaml"
            ),
            "--full-production",
            "--preflight",
        ]
        good = subprocess.run(
            [*base, "--max-updates", "1794000"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert good.returncode == 0, f"{good.stdout}\n{good.stderr}"
        payload = json.loads(good.stdout.splitlines()[-1])
        assert payload == {
            "artifacts_created": False,
            "configured_optimizer_updates": 1_794_000,
            "identity": payload["identity"],
            "model": "dp_cnn",
            "rollout_during_training": False,
            "status": "full-production-preflight",
            "training_mode": "full_production",
        }
        assert payload["identity"]["optimizer_updates"] == 1_794_000
        assert not output.exists()
        assert json.loads(index_path.read_text(encoding="utf-8"))["artifacts"] == {}

        bad = subprocess.run(
            [*base, "--max-updates", "100000"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert bad.returncode == 1
        assert (
            bad.stdout
            == "FAIL CLOSED: --full-production requires --max-updates 1794000 for dp_cnn\n"
        )
        assert not output.exists()
        assert json.loads(index_path.read_text(encoding="utf-8"))["artifacts"] == {}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_launcher_keyboard_interrupt_cleanup_retry_and_immutable_output(
    canonical_test_root: Path,
) -> None:
    root = canonical_test_root
    try:
        store = _store(root, "fixture")
        artifact_root = root / "artifacts"
        artifact_root.mkdir()
        index_path = root / "index.json"
        index_path.write_text('{"schema":1,"artifacts":{}}\n', encoding="utf-8")
        driver = """
import hashlib, json, sys
from pathlib import Path
import so101_pusht_benchmark.training.launcher as launcher
from so101_pusht_benchmark.training.artifacts import ArtifactError, ArtifactIndex
from so101_pusht_benchmark.training.launcher import TrainingLaunch
root, store, output, index_path = map(Path, sys.argv[1:5])
index = ArtifactIndex(index_path, root)
original = launcher.run_workspace_one_batch
def cancel(*args, **kwargs): raise KeyboardInterrupt('injected cancellation')
launcher.run_workspace_one_batch = cancel
try:
    launcher.launch_training(store, output, index, TrainingLaunch(0, 'cancel', model='dp_cnn', smoke_mode='fixture'))
except KeyboardInterrupt:
    pass
else:
    raise SystemExit('cancellation unexpectedly succeeded')
assert not output.exists()
assert not list(root.glob('.final.tmp-*'))
assert json.loads(index_path.read_text())['artifacts'] == {}
launcher.run_workspace_one_batch = original
original_anchor = index.anchor_checkpoint
def fail_anchor(*args, **kwargs): raise RuntimeError('injected index failure')
index.anchor_checkpoint = fail_anchor
try:
    launcher.launch_training(store, output, index, TrainingLaunch(0, 'anchor-fail', model='dp_cnn', smoke_mode='fixture'))
except RuntimeError as exc:
    assert 'injected index failure' in str(exc)
else:
    raise SystemExit('index failure unexpectedly succeeded')
assert not output.exists()
assert not list(root.glob('.final.tmp-*'))
assert json.loads(index_path.read_text())['artifacts'] == {}
index.anchor_checkpoint = original_anchor
checkpoint = launcher.launch_training(store, output, index, TrainingLaunch(0, 'retry', model='dp_cnn', smoke_mode='fixture'))
def digest_tree():
    return [(str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(root.rglob('*')) if path.is_file()]
before = digest_tree()
try:
    launcher.launch_training(store, output, index, TrainingLaunch(0, 'second', model='dp_cnn', smoke_mode='fixture'))
except ArtifactError as exc:
    assert 'output already exists' in str(exc)
else:
    raise SystemExit('immutable output unexpectedly accepted')
assert digest_tree() == before
print('TRANSACTION_JSON=' + json.dumps({'checkpoint': str(checkpoint), 'clean_retry': True, 'immutable': True}, sort_keys=True))
"""
        environment = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WANDB_MODE": "offline",
        }
        result = subprocess.run(
            [
                str(_paper_python()),
                "-c",
                driver,
                str(artifact_root),
                str(store),
                str(artifact_root / "final"),
                str(index_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert '"clean_retry": true' in result.stdout
        assert '"immutable": true' in result.stdout
    finally:
        shutil.rmtree(root, ignore_errors=True)
