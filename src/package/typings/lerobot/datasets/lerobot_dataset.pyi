from pathlib import Path

class LeRobotDataset:
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict[str, object],
        root: str | Path | None = ...,
        robot_type: str | None = ...,
        use_videos: bool = ...,
        tolerance_s: float = ...,
        image_writer_processes: int = ...,
        image_writer_threads: int = ...,
        video_backend: str | None = ...,
        batch_encoding_size: int = ...,
        vcodec: str = ...,
        metadata_buffer_size: int = ...,
        streaming_encoding: bool = ...,
        encoder_queue_maxsize: int = ...,
        encoder_threads: int | None = ...,
    ) -> LeRobotDataset: ...
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = ...,
        download_videos: bool = ...,
    ) -> None: ...
    def add_frame(self, frame: dict[str, object]) -> None: ...
    def save_episode(
        self,
        episode_data: dict[str, object] | None = ...,
        parallel_encoding: bool = ...,
    ) -> None: ...
    def finalize(self) -> None: ...
    def __len__(self) -> int: ...
