from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class BaseImageRunner:
    output_dir: str

    def __init__(self, output_dir: str) -> None: ...

    def run(self, policy: BaseImagePolicy) -> dict[str, object]: ...
