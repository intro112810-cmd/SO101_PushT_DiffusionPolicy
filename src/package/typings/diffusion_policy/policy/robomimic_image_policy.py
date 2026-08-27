from torch import nn

from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class RobomimicImagePolicy(BaseImagePolicy):
    model: object
    nets: nn.ModuleDict
    config: object
