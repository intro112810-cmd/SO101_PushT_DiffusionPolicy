class SOFollowerRobotConfig:
    def __init__(
        self,
        *,
        port: str,
        id: str,
        cameras: dict[str, object],
        use_degrees: bool,
    ) -> None: ...
