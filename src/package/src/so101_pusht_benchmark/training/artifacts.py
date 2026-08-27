"""Fail-closed digest index for generated model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import json
import os
import pwd
import secrets
import stat
from pathlib import Path
from typing import Literal, cast
from .budgets import APPROVED_OPTIMIZER_UPDATES, local_optimizer_updates

ReceiptStage = Literal["training", "bundle", "evaluation"]
_RECEIPT_SCHEMA = "so101-pusht-production-stage-receipt-v2"
_STAGE_PREDECESSOR: dict[ReceiptStage, ReceiptStage | None] = {
    "training": None,
    "bundle": "training",
    "evaluation": "bundle",
}


class ArtifactError(RuntimeError):
    """Raised when a generated artifact cannot be trusted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactScope:
    config: Path | None = None
    simulation_probe: bool = False
    smoke_mode: Literal["fixture", "production"] | None = None
    training_mode: Literal["full_production"] | None = None
    identity: dict[str, object] | None = None
    production_receipt: Path | None = None
    training_log: Path | None = None


@dataclass(frozen=True, slots=True)
class BundleFiles:
    bundle: Path
    config: Path
    normalizer: Path
    manifest: Path | None = None


def require_production_artifact(record: Mapping[str, object], *, operation: str) -> None:
    """Allow only a completed full-training checkpoint/bundle for final operations."""
    identity = record.get("identity")
    updates: object = None
    model: object = None
    if isinstance(identity, dict):
        typed_identity = cast("dict[str, object]", identity)
        updates = typed_identity.get("optimizer_updates")
        model = typed_identity.get("model")
    expected_status = {
        "bundle export": "full_training_complete",
        "evaluation": "full_training_bundle_ready",
    }.get(operation)
    local_budget = os.environ.get("PUSHT_LOCAL_BUDGET") == "1"
    if local_budget:
        try:
            allowed_updates = local_optimizer_updates(model)
        except ValueError:
            allowed_updates = None
    else:
        allowed_updates = APPROVED_OPTIMIZER_UPDATES.get(model)
    if (
        expected_status is None
        or record.get("deployment_scope") != "simulation_only"
        or record.get("training_eligible") is not True
        or record.get("comparison_eligible") is not False
        or record.get("result_status") != expected_status
        or not isinstance(model, str)
        or updates != allowed_updates
    ):
        raise ArtifactError(
            f"{operation} requires a completed approved-budget full-production artifact; "
            "smoke checkpoints are non-final"
        )


class ArtifactIndex:
    """Read and atomically update the tracked model artifact index."""

    def __init__(self, path: Path, artifact_root: Path) -> None:
        self.path = path.absolute()
        self.artifact_root = artifact_root.resolve()
        root_binding = hashlib.sha256(str(self.artifact_root).encode()).hexdigest()
        # Resolve through the OS account database, never a caller-controlled HOME value.
        account_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        self.receipt_root = (
            account_home / ".local/state/so101-pusht-benchmark/producer-receipts" / root_binding
        )

    @staticmethod
    def _canonical_json(value: object) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _require_owner_mode(path: Path, mode: int, label: str) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise ArtifactError(f"missing producer receipt {label}: {path}") from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or (label == "root" and not stat.S_ISDIR(info.st_mode))
            or (label != "root" and not stat.S_ISREG(info.st_mode))
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != mode
        ):
            raise ArtifactError(f"producer receipt {label} ownership or mode is unsafe")

    def _receipt_path(self, artifact_id: str, stage: ReceiptStage = "training") -> Path:
        name = hashlib.sha256(f"{artifact_id}:{stage}".encode()).hexdigest()
        return self.receipt_root / f"{name}.json"

    def _reject_receipt_symlinks(self) -> None:
        current = Path(self.receipt_root.anchor)
        for part in self.receipt_root.parts[1:]:
            current /= part
            try:
                info = current.lstat()
            except FileNotFoundError:
                break
            if stat.S_ISLNK(info.st_mode):
                raise ArtifactError("producer receipt path contains a symlink")
            if not stat.S_ISDIR(info.st_mode):
                raise ArtifactError("producer receipt parent is not a directory")
            if info.st_uid not in {0, os.getuid()} or stat.S_IMODE(info.st_mode) & 0o022:
                raise ArtifactError("producer receipt parent ownership or mode is unsafe")

    @staticmethod
    def _read_key(key_path: Path) -> bytes:
        key = key_path.read_bytes()
        if len(key) != 32:
            raise ArtifactError("producer receipt key is malformed")
        return key

    def _initialize_receipt_store(self) -> Path:
        account_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        current = account_home
        for part in self.receipt_root.relative_to(account_home).parts:
            current /= part
            try:
                info = current.lstat()
            except FileNotFoundError:
                current.mkdir(mode=0o700)
                continue
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ArtifactError("producer receipt path is unsafe")
            if info.st_uid != os.getuid():
                raise ArtifactError("producer receipt path owner is unsafe")
            current.chmod(0o700)
        self._reject_receipt_symlinks()
        self._require_owner_mode(self.receipt_root, 0o700, "root")
        key_path = self.receipt_root / "producer.key"
        try:
            descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(secrets.token_bytes(32))
        self._require_owner_mode(key_path, 0o400, "key")
        return key_path

    def _read_receipt_document(self, path: Path, key: bytes) -> dict[str, object]:
        self._require_owner_mode(path, 0o400, "file")
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError("invalid producer receipt") from exc
        if not isinstance(raw, dict):
            raise ArtifactError("invalid producer receipt")
        signed = cast("dict[str, object]", raw)
        document = dict(signed)
        signature = document.pop("hmac_sha256", None)
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature, hmac.digest(key, self._canonical_json(document), "sha256").hex()
        ):
            raise ArtifactError("producer receipt authentication failed")
        if set(document) != {
            "schema",
            "stage",
            "artifact_root",
            "artifact_index",
            "artifact_id",
            "predecessor_sha256",
            "record",
            "contract",
        }:
            raise ArtifactError("producer receipt binding mismatch")
        stage = document.get("stage")
        artifact_id = document.get("artifact_id")
        if (
            document.get("schema") != _RECEIPT_SCHEMA
            or stage not in _STAGE_PREDECESSOR
            or not isinstance(artifact_id, str)
            or path != self._receipt_path(artifact_id, stage)
            or document.get("artifact_root") != str(self.artifact_root)
            or document.get("artifact_index") != str(self.artifact_root / "artifact-index.json")
            or not isinstance(document.get("record"), dict)
            or not isinstance(document.get("contract"), dict)
        ):
            raise ArtifactError("producer receipt binding mismatch")
        return document

    def _write_producer_receipt(
        self,
        artifact_id: str,
        stage: ReceiptStage,
        record: dict[str, object],
        contract: dict[str, object],
    ) -> Path:
        key_path = self._initialize_receipt_store()
        receipt_path = self._receipt_path(artifact_id, stage)
        predecessor = _STAGE_PREDECESSOR[stage]
        predecessor_digest = (
            None
            if predecessor is None
            else sha256_file(self._receipt_path(artifact_id, predecessor))
        )
        payload = {
            "schema": _RECEIPT_SCHEMA,
            "stage": stage,
            "artifact_root": str(self.artifact_root),
            "artifact_index": str(self.artifact_root / "artifact-index.json"),
            "artifact_id": artifact_id,
            "predecessor_sha256": predecessor_digest,
            "record": record,
            "contract": contract,
        }
        signature = hmac.digest(self._read_key(key_path), self._canonical_json(payload), "sha256")
        document = {**payload, "hmac_sha256": signature.hex()}
        try:
            descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        except FileExistsError as exc:
            raise ArtifactError(
                f"producer receipt identity already exists: {artifact_id}:{stage}"
            ) from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(self._canonical_json(document) + b"\n")
        return receipt_path

    def stage_receipt_path(self, artifact_id: str, stage: ReceiptStage) -> Path:
        """Return the canonical owner-protected receipt path for a production stage."""
        return self._receipt_path(artifact_id, stage)

    def authenticate_stage(self, artifact_id: str, stage: ReceiptStage) -> dict[str, object]:
        """Authenticate a complete immutable producer-stage snapshot and its chain."""
        canonical_index = self.artifact_root / "artifact-index.json"
        if self.path != canonical_index or self.path.is_symlink():
            raise ArtifactError("production stage requires the canonical artifact index")
        self._reject_receipt_symlinks()
        self._require_owner_mode(self.receipt_root, 0o700, "root")
        key_path = self.receipt_root / "producer.key"
        self._require_owner_mode(key_path, 0o400, "key")
        key = self._read_key(key_path)
        documents: dict[tuple[str, ReceiptStage], dict[str, object]] = {}
        candidates: list[Path] = []
        for entry in self.receipt_root.iterdir():
            if entry.name == "producer.key":
                continue
            if entry.suffix != ".json":
                raise ArtifactError("unexpected producer receipt store entry")
            candidates.append(entry)
        for candidate in candidates:
            document = self._read_receipt_document(candidate, key)
            candidate_id = cast("str", document["artifact_id"])
            candidate_stage = cast("ReceiptStage", document["stage"])
            identity = (candidate_id, candidate_stage)
            if identity in documents:
                raise ArtifactError("stale or duplicated producer stage identity")
            documents[identity] = document
        target = documents.get((artifact_id, stage))
        if target is None:
            raise ArtifactError(f"missing producer receipt stage: {artifact_id}:{stage}")
        current = target
        current_stage = stage
        while True:
            predecessor = _STAGE_PREDECESSOR[current_stage]
            if predecessor is None:
                if current.get("predecessor_sha256") is not None:
                    raise ArtifactError("producer receipt chain mismatch")
                break
            prior = documents.get((artifact_id, predecessor))
            prior_path = self._receipt_path(artifact_id, predecessor)
            if prior is None or current.get("predecessor_sha256") != sha256_file(prior_path):
                raise ArtifactError("producer receipt chain mismatch")
            current = prior
            current_stage = predecessor
        signed_record = cast("dict[str, object]", target["record"])
        if signed_record != self.record(artifact_id):
            raise ArtifactError("artifact index differs from producer receipt")
        contract = cast("dict[str, object]", target["contract"])
        inventory = contract.get("file_inventory")
        if not isinstance(inventory, list):
            raise ArtifactError("producer receipt file inventory is invalid")
        for item in cast("list[object]", inventory):
            if not isinstance(item, dict):
                raise ArtifactError("producer receipt file inventory is invalid")
            entry = cast("dict[str, object]", item)
            if set(entry) != {"label", "path", "sha256"}:
                raise ArtifactError("producer receipt file inventory is invalid")
            label = entry.get("label")
            if not isinstance(label, str):
                raise ArtifactError("producer receipt file inventory is invalid")
            path = self.verify(artifact_id, label)
            self._require_owner_mode(path, 0o400, "artifact")
            if entry.get("path") != signed_record.get(f"{label}_path") or entry.get(
                "sha256"
            ) != sha256_file(path):
                raise ArtifactError("producer receipt file inventory mismatch")
        return contract

    def require_trusted_production_checkpoint(self, artifact_id: str) -> tuple[Path, Path]:
        """Authenticate immutable producer output before unsafe checkpoint loading."""
        record = self.record(artifact_id)
        require_production_artifact(record, operation="bundle export")
        contract = self.authenticate_stage(artifact_id, "training")
        if contract.get("result_status") != "full_training_complete" or contract.get(
            "identity"
        ) != record.get("identity"):
            raise ArtifactError("producer training receipt contract mismatch")
        checkpoint = self.verify(artifact_id, "checkpoint")
        config = self.verify(artifact_id, "config")
        production_receipt = self.verify(artifact_id, "production_receipt")
        self.verify(artifact_id, "training_log")
        try:
            training: object = json.loads(production_receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError("invalid full-production training receipt") from exc
        identity = record.get("identity")
        identity_model = (
            cast("dict[str, object]", identity).get("model")
            if isinstance(identity, dict)
            else None
        )
        expected_updates: object = None
        if isinstance(identity_model, str):
            if os.environ.get("PUSHT_LOCAL_BUDGET") == "1":
                try:
                    expected_updates = local_optimizer_updates(identity_model)
                except ValueError:
                    expected_updates = None
            else:
                expected_updates = APPROVED_OPTIMIZER_UPDATES.get(identity_model)
        expected_training: dict[str, object] = {
            "schema": "pusht-so100-full-training-v1",
            "model": identity_model,
            "training_mode": "full_production",
            "configured_optimizer_updates": expected_updates,
            "executed_optimizer_updates": expected_updates,
            "rollout_during_training": False,
            "completed": True,
            "identity": identity,
        }
        if training != expected_training:
            raise ArtifactError("full-production training receipt identity mismatch")
        return checkpoint, config

    def _safe_file(self, path: Path) -> Path:
        lexical = path.absolute()
        current = Path(lexical.anchor)
        for part in lexical.parts[1:]:
            current /= part
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError as exc:
                raise ArtifactError(f"missing artifact: {path}") from exc
            if stat.S_ISLNK(mode):
                raise ArtifactError(f"symlink artifact path is forbidden: {path}")
        resolved = lexical.resolve()
        if resolved == self.artifact_root or self.artifact_root not in resolved.parents:
            raise ArtifactError(f"artifact is outside artifact root: {path}")
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise ArtifactError(f"artifact is not a regular file: {path}")
        return resolved

    def create_output_directory(self, path: Path, allow_existing: bool = False) -> Path:
        """Create a new output directory without following links or leaving the artifact root.

        allow_existing=True permits the final path to already exist (resume mode:
        the staging directory holds a preserved checkpoint from a crashed run).
        """
        lexical = path.absolute()
        root = self.artifact_root
        if lexical == root or root not in lexical.parents:
            raise ArtifactError(f"output is outside artifact root: {path}")
        relative = lexical.relative_to(root)
        current = root
        if root.is_symlink() or not stat.S_ISDIR(root.stat().st_mode):
            raise ArtifactError("artifact root must be a real directory")
        for part in relative.parts:
            current /= part
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError:
                current.mkdir()
                continue
            except NotADirectoryError as exc:
                raise ArtifactError(f"output parent is not a directory: {current.parent}") from exc
            if current == lexical:
                if not allow_existing:
                    raise ArtifactError(f"output already exists: {path}")
                continue
            if stat.S_ISLNK(mode):
                raise ArtifactError(f"symlink output path is forbidden: {path}")
            if not stat.S_ISDIR(mode):
                raise ArtifactError(f"output parent is not a directory: {current}")
        if current != lexical:
            raise ArtifactError("output path resolution failed")
        return current

    def _load(self) -> dict[str, object]:
        try:
            if self.path.is_symlink() or not stat.S_ISREG(self.path.stat().st_mode):
                raise ArtifactError("artifact index must be a regular non-symlink file")
            raw: object = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"invalid artifact index: {self.path}") from exc
        if not isinstance(raw, dict):
            raise ArtifactError("artifact index must be an object")
        data = cast("dict[str, object]", raw)
        if data.get("schema") != 1 or not isinstance(data.get("artifacts"), dict):
            raise ArtifactError("unsupported artifact index schema")
        return data

    def merge_record(self, artifact_id: str, values: dict[str, object]) -> None:
        """Merge one record under a process-safe index lock."""
        if not artifact_id or artifact_id in {".", ".."}:
            raise ArtifactError("invalid artifact id")
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "r+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = self._load()
                records = cast("dict[str, object]", data["artifacts"])
                previous = records.get(artifact_id, {})
                if not isinstance(previous, dict):
                    raise ArtifactError("invalid artifact record")
                records[artifact_id] = {**cast("dict[str, object]", previous), **values}
                temporary = self.path.with_name(
                    f".{self.path.name}.{os.getpid()}.tmp"
                )
                if temporary.exists() or temporary.is_symlink():
                    raise ArtifactError(f"artifact index staging already exists: {temporary}")
                try:
                    temporary.write_text(
                        json.dumps(data, indent=2, sort_keys=False) + "\n",
                        encoding="utf-8",
                    )
                    temporary.replace(self.path)
                finally:
                    temporary.unlink(missing_ok=True)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _fields(self, label: str, path: Path) -> dict[str, object]:
        safe = self._safe_file(path)
        return {
            f"{label}_path": safe.relative_to(self.artifact_root).as_posix(),
            f"{label}_sha256": sha256_file(safe),
        }

    @staticmethod
    def _inventory(
        record: Mapping[str, object], labels: tuple[str, ...]
    ) -> list[dict[str, object]]:
        inventory: list[dict[str, object]] = []
        for label in labels:
            path = record.get(f"{label}_path")
            digest = record.get(f"{label}_sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                raise ArtifactError(f"production {label} inventory is incomplete")
            inventory.append({"label": label, "path": path, "sha256": digest})
        return inventory

    @staticmethod
    def _scope(
        simulation_probe: bool,
        smoke_mode: Literal["fixture", "production"] | None = None,
        training_mode: Literal["full_production"] | None = None,
    ) -> dict[str, object]:
        if smoke_mode == "fixture":
            return {
                "deployment_scope": "simulation_only",
                "training_eligible": False,
                "comparison_eligible": False,
                "result_status": "ineligible_fixture",
            }
        if smoke_mode == "production":
            return {
                "deployment_scope": "simulation_only",
                "training_eligible": False,
                "comparison_eligible": False,
                "result_status": "production_smoke_complete_nonfinal",
            }
        if training_mode == "full_production":
            return {
                "deployment_scope": "simulation_only",
                "training_eligible": True,
                "comparison_eligible": False,
                "result_status": "full_training_complete",
            }
        return {
            "deployment_scope": "simulation_only",
            "training_eligible": not simulation_probe,
            "comparison_eligible": False,
            "result_status": "synthetic_pipeline_probe" if simulation_probe else "candidate",
        }

    def anchor_checkpoint(
        self,
        artifact_id: str,
        checkpoint: Path,
        options: ArtifactScope | None = None,
    ) -> None:
        selected = ArtifactScope() if options is None else options
        fields = self._fields("checkpoint", checkpoint)
        if selected.config is not None:
            fields.update(self._fields("config", selected.config))
        if selected.identity is not None:
            fields["identity"] = selected.identity
        if selected.production_receipt is not None:
            fields.update(self._fields("production_receipt", selected.production_receipt))
        record = {
            **fields,
            **self._scope(selected.simulation_probe, selected.smoke_mode, selected.training_mode),
        }
        producer_receipt: Path | None = None
        if selected.training_mode == "full_production":
            if selected.production_receipt is None or selected.training_log is None:
                raise ArtifactError(
                    "full production requires producer-owned training receipt and log"
                )
            record.update(self._fields("training_log", selected.training_log))
            for immutable in (
                checkpoint,
                selected.config,
                selected.production_receipt,
                selected.training_log,
            ):
                if immutable is None:
                    raise ArtifactError("full production trust chain is incomplete")
                immutable.chmod(0o400)
            producer_receipt = self._write_producer_receipt(
                artifact_id,
                "training",
                record,
                {
                    "result_status": "full_training_complete",
                    "deployment_scope": "simulation_only",
                    "training_eligible": True,
                    "comparison_eligible": False,
                    "identity": record.get("identity"),
                    "file_inventory": self._inventory(
                        record,
                        ("checkpoint", "config", "production_receipt", "training_log"),
                    ),
                },
            )
        try:
            self.merge_record(artifact_id, record)
        except BaseException:
            if producer_receipt is not None:
                producer_receipt.unlink(missing_ok=True)
            raise

    def anchor_bundle(
        self,
        artifact_id: str,
        files: BundleFiles,
        options: ArtifactScope | None = None,
    ) -> None:
        selected = ArtifactScope() if options is None else options
        fields = {
            **self._fields("bundle", files.bundle),
            **self._fields("config", files.config),
            **self._fields("normalizer", files.normalizer),
        }
        if files.manifest is not None:
            fields.update(self._fields("manifest", files.manifest))
        if selected.identity is not None:
            fields["identity"] = selected.identity
        scope = self._scope(selected.simulation_probe, selected.smoke_mode, selected.training_mode)
        if selected.training_mode != "full_production":
            self.merge_record(artifact_id, {**fields, **scope})
            return
        self.authenticate_stage(artifact_id, "training")
        scope["result_status"] = "full_training_bundle_ready"
        current = self.record(artifact_id)
        record = {**current, **fields, **scope}
        immutable = (files.bundle, files.config, files.normalizer, files.manifest)
        for path in immutable:
            if path is None:
                raise ArtifactError("full-production bundle inventory is incomplete")
            path.chmod(0o400)
        producer_receipt = self._write_producer_receipt(
            artifact_id,
            "bundle",
            record,
            {
                "result_status": "full_training_bundle_ready",
                "deployment_scope": "simulation_only",
                "training_eligible": True,
                "comparison_eligible": False,
                "identity": record.get("identity"),
                "bundle_schema": 1,
                "file_inventory": self._inventory(
                    record,
                    ("checkpoint", "bundle", "config", "normalizer", "manifest"),
                ),
            },
        )
        try:
            self.merge_record(artifact_id, {**fields, **scope})
        except BaseException:
            producer_receipt.unlink(missing_ok=True)
            raise

    def anchor_evaluation(
        self,
        artifact_id: str,
        metrics: Path,
        *,
        identity: dict[str, object],
        failure_traces: Path | None = None,
    ) -> None:
        record = self.record(artifact_id)
        require_production_artifact(record, operation="evaluation")
        self.authenticate_stage(artifact_id, "bundle")
        traces = (
            metrics.with_name("failure_traces.json") if failure_traces is None else failure_traces
        )
        fields = {
            **self._fields("metrics", metrics),
            **self._fields("failure_traces", traces),
            "identity": identity,
            "comparison_eligible": True,
            "result_status": "anchored_final_evaluation",
        }
        final_record = {**record, **fields}
        try:
            raw: object = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError("final evaluation metrics are invalid") from exc
        if not isinstance(raw, dict):
            raise ArtifactError("final evaluation metrics are invalid")
        document = cast("dict[str, object]", raw)
        for path in (metrics, traces):
            path.chmod(0o400)
        producer_receipt = self._write_producer_receipt(
            artifact_id,
            "evaluation",
            final_record,
            {
                "result_status": "anchored_final_evaluation",
                "deployment_scope": "simulation_only",
                "training_eligible": True,
                "comparison_eligible": True,
                "identity": identity,
                "metric_schema": document.get("metric_schema"),
                "evaluation_seeds": document.get("evaluation_seeds"),
                "step_cap": document.get("step_cap"),
                "fps": document.get("fps"),
                "file_inventory": self._inventory(
                    final_record,
                    (
                        "checkpoint",
                        "bundle",
                        "config",
                        "normalizer",
                        "manifest",
                        "metrics",
                        "failure_traces",
                    ),
                ),
            },
        )
        try:
            self.merge_record(artifact_id, fields)
        except BaseException:
            producer_receipt.unlink(missing_ok=True)
            raise

    def record(self, artifact_id: str) -> dict[str, object]:
        records = cast("dict[str, object]", self._load()["artifacts"])
        record = records.get(artifact_id)
        if not isinstance(record, dict):
            raise ArtifactError(f"artifact is not anchored: {artifact_id}")
        return cast("dict[str, object]", record)

    def verify(self, artifact_id: str, label: str) -> Path:
        record = self.record(artifact_id)
        relative = record.get(f"{label}_path")
        expected = record.get(f"{label}_sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ArtifactError(f"invalid anchored {label} path")
        path = self._safe_file(self.artifact_root / relative)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ArtifactError(f"{label} digest mismatch")
        return path