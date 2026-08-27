from __future__ import annotations

from typing import TypeAlias

from torch import Tensor
from torch.utils.data import Dataset

from diffusion_policy.model.common.normalizer import LinearNormalizer

Sample: TypeAlias = dict[str, Tensor | dict[str, Tensor]]


class BaseImageDataset(Dataset[Sample]):
    def __init__(self) -> None: ...

    def get_validation_dataset(self) -> BaseImageDataset: ...

    def get_normalizer(self, mode: str = "limits") -> LinearNormalizer: ...

    def get_all_actions(self) -> Tensor: ...

    def __len__(self) -> int:
        """Return the number of sequence samples."""

    def __getitem__(self, idx: int) -> Sample:
        """Return one policy observation and action sequence."""
