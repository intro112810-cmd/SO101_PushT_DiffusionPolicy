"""Deterministic non-actuating pipeline from physical captures to shadow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Final

_ROOT: Final = Path(__file__).resolve().parents[3]
_SCRIPTS: Final = _ROOT / "scripts"


@dataclass(frozen=True, slots=True)
class PostCapturePaths:
    run_root: Path
    profile_template: Path
    acquisition_authority: Path
    acquisition_signature: Path
    trust_anchor: Path
    intrinsic_evidence: Path
    table_fit_a: Path
    table_fit_b: Path
    table_fit_c: Path
    checkpoint_held_a: Path
    checkpoint_held_b: Path
    camera_policy: Path
    private_key: Path
    joint_corpus: Path
    joint_policy: Path
    lineage: Path
    artifact_root: Path
    frame: Path
    samples: Path
    measured_square_mm: float = 25.0

    @classmethod
    def fixture(cls, root: Path) -> PostCapturePaths:
        """Build deterministic paths for command-contract tests."""
        inputs = root / "inputs"
        return cls(
            root,
            inputs / "profile.yaml",
            inputs / "acquisition.json",
            inputs / "acquisition.sig",
            inputs / "anchor.pem",
            inputs / "intrinsics",
            inputs / "fit-a.mkv",
            inputs / "fit-b.mkv",
            inputs / "fit-c.mkv",
            inputs / "held-a.mkv",
            inputs / "held-b.mkv",
            inputs / "camera-policy.yaml",
            inputs / "private-key.pem",
            inputs / "joint-corpus.json",
            inputs / "joint-policy.yaml",
            inputs / "lineage.json",
            inputs / "artifacts",
            inputs / "frame.png",
            inputs / "samples.json",
        )

    @property
    def camera_output_dir(self) -> Path:
        return self.run_root / "camera-corpus"

    @property
    def camera_authority_dir(self) -> Path:
        return self.run_root / "camera-authority"

    @property
    def camera_identity(self) -> Path:
        return self.camera_authority_dir / "live-identity.json"

    @property
    def camera_corpus_authority(self) -> Path:
        return self.camera_authority_dir / "camera-corpus-authority.json"

    @property
    def joint_authority_dir(self) -> Path:
        return self.run_root / "joint-authority"

    @property
    def joint_corpus_authority(self) -> Path:
        return self.joint_authority_dir / "joint-corpus-authority.json"

    @property
    def camera_receipt(self) -> Path:
        return self.run_root / "camera-receipt.json"

    @property
    def joint_receipt(self) -> Path:
        return self.run_root / "joint-receipt.json"

    @property
    def bound_profile(self) -> Path:
        return self.run_root / "hardware-profile-bound.yaml"

    @property
    def shadow_output(self) -> Path:
        return self.run_root / "physical-shadow.json"

    def outputs(self) -> tuple[Path, ...]:
        return (
            self.camera_output_dir,
            self.camera_authority_dir,
            self.joint_authority_dir,
            self.camera_receipt,
            self.joint_receipt,
            self.bound_profile,
            self.shadow_output,
        )


def build_commands(paths: PostCapturePaths, *, python: str) -> tuple[list[str], ...]:
    """Return the exact non-actuating command chain in dependency order."""
    camera_corpus = paths.camera_output_dir / "corpus.json"
    lineage_authority = (
        json.loads(paths.lineage.read_text(encoding="utf-8")).get("authority_digest", "")
        if paths.lineage.is_file()
        else ""
    )
    return (
        [
            python,
            str(_SCRIPTS / "build_camera_corpus_from_video.py"),
            "--profile",
            str(paths.profile_template),
            "--acquisition-authority",
            str(paths.acquisition_authority),
            "--authority-signature",
            str(paths.acquisition_signature),
            "--trust-anchor",
            str(paths.trust_anchor),
            "--intrinsic-evidence",
            str(paths.intrinsic_evidence),
            "--table-fit-a",
            str(paths.table_fit_a),
            "--table-fit-b",
            str(paths.table_fit_b),
            "--table-fit-c",
            str(paths.table_fit_c),
            "--checkpoint-held-a",
            str(paths.checkpoint_held_a),
            "--checkpoint-held-b",
            str(paths.checkpoint_held_b),
            "--output-dir",
            str(paths.camera_output_dir),
            "--measured-square-mm",
            str(paths.measured_square_mm),
        ],
        [
            python,
            str(_SCRIPTS / "issue_camera_corpus_authority_offline.py"),
            "--base-authority",
            str(paths.acquisition_authority),
            "--base-signature",
            str(paths.acquisition_signature),
            "--policy",
            str(paths.camera_policy),
            "--trust-anchor",
            str(paths.trust_anchor),
            "--private-key",
            str(paths.private_key),
            "--corpus",
            str(camera_corpus),
            "--output-dir",
            str(paths.camera_authority_dir),
            "--approval-id",
            "tomorrow-camera-exact-corpus",
        ],
        [
            python,
            str(_SCRIPTS / "prepare_joint_corpus_authority.py"),
            "issue",
            "--corpus",
            str(paths.joint_corpus),
            "--trust-anchor",
            str(paths.trust_anchor),
            "--private-key",
            str(paths.private_key),
            "--approval-id",
            "tomorrow-joint-exact-corpus",
            "--output-dir",
            str(paths.joint_authority_dir),
            "--output",
            str(paths.joint_corpus_authority),
        ],
        [
            python,
            str(_SCRIPTS / "audit_camera_registration.py"),
            "--mode",
            "physical",
            "--corpus",
            str(camera_corpus),
            "--policy",
            str(paths.camera_policy),
            "--identity",
            str(paths.camera_identity),
            "--corpus-authority",
            str(paths.camera_corpus_authority),
            "--trust-anchor",
            str(paths.trust_anchor),
            "--output",
            str(paths.camera_receipt),
        ],
        [
            python,
            str(_SCRIPTS / "audit_joint_equivalence_read_only.py"),
            "--governed-physical",
            "--corpus",
            str(paths.joint_corpus),
            "--policy",
            str(paths.joint_policy),
            "--corpus-authority",
            str(paths.joint_corpus_authority),
            "--trust-anchor",
            str(paths.trust_anchor),
            "--output",
            str(paths.joint_receipt),
        ],
        [
            python,
            str(_SCRIPTS / "bind_hardware_profile.py"),
            "--template",
            str(paths.profile_template),
            "--lineage",
            str(paths.lineage),
            "--joint-receipt",
            str(paths.joint_receipt),
            "--camera-receipt",
            str(paths.camera_receipt),
            "--policy",
            str(paths.joint_policy),
            "--trust-anchor",
            str(paths.trust_anchor),
            "--output",
            str(paths.bound_profile),
            "--action-bridge-audited",
        ],
        [
            python,
            str(_SCRIPTS / "run_real_shadow_inference.py"),
            "--artifact-root",
            str(paths.artifact_root),
            "--model",
            "dp_cnn",
            "--artifact",
            "local-dp_cnn-recovered-v3-seed0",
            "--frame",
            str(paths.frame),
            "--samples",
            str(paths.samples),
            "--lineage",
            str(paths.lineage),
            "--lineage-authority-digest",
            str(lineage_authority),
            "--joint",
            str(paths.joint_receipt),
            "--camera",
            str(paths.camera_receipt),
            "--hardware-profile",
            str(paths.bound_profile),
            "--output",
            str(paths.shadow_output),
        ],
    )


def execute_pipeline(paths: PostCapturePaths, *, python: str) -> None:
    """Run every gate once; refuse stale outputs and stop on first failure."""
    stale = [path for path in paths.outputs() if path.exists()]
    if stale:
        raise ValueError("fresh run root required; existing outputs: " + ", ".join(map(str, stale)))
    paths.run_root.mkdir(parents=True, exist_ok=True)
    for command in build_commands(paths, python=python):
        subprocess.run(command, check=True, cwd=_ROOT)
