"""Fresh digest-chained cycle package provider for production bounded rollout."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import cast, Protocol
from .authorization import AuthorizationToken
from .bounded_execution import DurableIntentLedger
from .physical_ik_replay import parse_physical_ik_proposal
from .production_bounded import ProductionCycle
from .production_evidence import DirectBusEvidenceConfig, DirectBusEvidenceProvider, ReadbackRobot
from .production_frame import OpenCvPostFrameSource
from .production_single_step import ProductionSingleStepRuntime
from .rollout_codes import RolloutCode, RolloutViolation
from .writer import DirectBusRobot

FILE_CYCLE_PROVIDER_DIGEST = hashlib.sha256(
    b"so101-production-digest-chained-file-cycle-provider-v1"
).hexdigest()


class CyclePackageBuilder(Protocol):
    @property
    def digest(self) -> str: ...
    def build(self, index: int, previous_evidence_digest: str, output: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class CommandCyclePackageBuilder:
    executable: Path
    expected_digest: str
    prefix_arguments: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        actual = hashlib.sha256(self.executable.read_bytes()).hexdigest()
        if actual != self.expected_digest:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "cycle builder executable drift")
        return actual

    def build(self, index: int, previous_evidence_digest: str, output: Path) -> None:
        _ = self.digest
        command = [
            str(self.executable),
            *self.prefix_arguments,
            "--cycle-index",
            str(index),
            "--previous-evidence-digest",
            previous_evidence_digest,
            "--output",
            str(output),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0 or not output.is_file():
            raise RolloutViolation(RolloutCode.F_PROVIDER_ERROR, "cycle builder failed")


@dataclass(frozen=True, slots=True)
class FileCycleProviderConfig:
    package_dir: Path
    output_dir: Path
    robot: DirectBusRobot
    readback_robot: ReadbackRobot
    provider_digest: str
    policy_digest: str
    camera_device: Path
    clock: Callable[[], float]
    builder: CyclePackageBuilder | None = None


class FileProductionCycleProvider:
    """Load cycle N only when requested and bind it to prior verified evidence."""

    def __init__(self, config: FileCycleProviderConfig) -> None:
        self._config = config
        self._intent = DurableIntentLedger(config.output_dir / "intents.jsonl")

    def next_cycle(self, index: int, previous_evidence_digest: str) -> ProductionCycle | None:
        path = self._config.package_dir / f"cycle-{index:03d}.json"
        if not path.exists():
            if self._config.builder is None:
                return None
            self._config.builder.build(index, previous_evidence_digest, path)
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RolloutViolation(RolloutCode.R_MISSING, "bounded cycle package") from exc
        if not isinstance(raw, dict):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "bounded cycle package")
        doc = cast("dict[str,object]", raw)
        expected = {
            "schema",
            "cycle",
            "previous_evidence_digest",
            "proposal",
            "token",
            "pre_sample_digests",
            "newer_than",
        }
        if (
            set(doc) != expected
            or doc.get("schema") != "production-bounded-cycle-v1"
            or doc.get("cycle") != index
            or doc.get("previous_evidence_digest") != previous_evidence_digest
        ):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "bounded cycle chain")
        proposal_raw = doc["proposal"]
        token_raw = doc["token"]
        pre_raw = doc["pre_sample_digests"]
        newer = doc["newer_than"]
        if (
            not isinstance(proposal_raw, Mapping)
            or not isinstance(token_raw, Mapping)
            or not isinstance(pre_raw, list)
            or isinstance(newer, bool)
            or not isinstance(newer, (int, float))
            or not math.isfinite(float(newer))
        ):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "bounded cycle content")
        proposal_map = cast("Mapping[str,object]", proposal_raw)
        declared = proposal_map.get("proposal_hash")
        if not isinstance(declared, str):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "bounded proposal hash")
        proposal = parse_physical_ik_proposal(
            {k: v for k, v in proposal_map.items() if k != "proposal_hash"}, declared_hash=declared
        )
        token_map = cast("Mapping[str,object]", token_raw)
        try:
            token = AuthorizationToken(
                str(token_map["token_id"]),
                str(token_map["proposal_hash"]),
                str(token_map["policy_digest"]),
                str(token_map["command_id"]),
                float(cast("int|float", token_map["valid_until"])),
                str(token_map["digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "bounded token") from exc
        if (
            token.policy_digest != self._config.policy_digest
            or token.proposal_hash != proposal.proposal_hash
        ):
            raise RolloutViolation(RolloutCode.R_HASH_MISMATCH, "bounded token policy/proposal")
        pre = frozenset(str(value) for value in cast("list[object]", pre_raw))
        evidence = DirectBusEvidenceProvider(
            DirectBusEvidenceConfig(
                self._config.readback_robot,
                self._config.provider_digest,
                token.command_id,
                proposal.proposal_hash,
                proposal.body_degrees,
                self._config.clock,
                OpenCvPostFrameSource(
                    self._config.camera_device,
                    self._config.output_dir / f"post-frame-{index:03d}.png",
                ),
            )
        )
        return ProductionCycle(
            token,
            proposal,
            pre,
            float(newer),
            ProductionSingleStepRuntime(
                self._config.robot, evidence, evidence, self._intent.append
            ),
        )
