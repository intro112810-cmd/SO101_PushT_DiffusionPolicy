import torch
from torch import Tensor
from torch import nn


class BaseImagePolicy(nn.Module):
    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def predict_action(self, obs_dict: dict[str, Tensor]) -> dict[str, Tensor]: ...

    def reset(self) -> None: ...
