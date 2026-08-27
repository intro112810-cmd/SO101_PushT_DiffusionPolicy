from pathlib import Path
from threading import Thread
from omegaconf import DictConfig

from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)


class TrainDiffusionUnetHybridWorkspace:
    cfg: DictConfig
    model: DiffusionUnetHybridImagePolicy
    ema_model: DiffusionUnetHybridImagePolicy | None
    global_step: int
    epoch: int
    _saving_thread: Thread | None

    def __init__(self, cfg: DictConfig, output_dir: str | None = None) -> None: ...
    def run(self) -> None: ...
    def get_checkpoint_path(self, tag: str = "latest") -> Path: ...
    def save_checkpoint(self, *, use_thread: bool = True) -> str: ...
    def load_checkpoint(self, path: Path | None = None, **kwargs: object) -> dict[str, object]: ...
