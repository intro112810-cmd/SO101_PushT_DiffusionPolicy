"""Training, tensor-only export, and live evaluation mechanics."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, cast

from .artifacts import ArtifactIndex

ResumeStage = Literal["training", "bundle", "evaluation"]


class _ResumeModule(Protocol):
    @staticmethod
    def validate_production_resume_artifact(
        index: ArtifactIndex,
        *,
        stage: ResumeStage,
        model: str,
        artifact_id: str,
        output: Path,
    ) -> dict[str, object]: ...


def validate_production_resume_artifact(
    index: ArtifactIndex,
    *,
    stage: ResumeStage,
    model: str,
    artifact_id: str,
    output: Path,
) -> dict[str, object]:
    """Invoke the package's read-only production resume validator."""
    module = cast("_ResumeModule", import_module(f"{__name__}.resume"))
    return module.validate_production_resume_artifact(
        index,
        stage=stage,
        model=model,
        artifact_id=artifact_id,
        output=output,
    )


__all__ = ["ResumeStage", "validate_production_resume_artifact"]
