"""Video loader: yields (frame_id, frame) from an MP4/AVI file."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Generator, Tuple


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    def frame_to_seconds(self, frame_id: int) -> float:
        return frame_id / self.fps if self.fps > 0 else 0.0


class VideoLoader:
    """Context-manager wrapper around cv2.VideoCapture."""

    def __init__(self, video_path: str):
        self.path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.info = VideoInfo(
            path=video_path,
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield (frame_id, BGR frame) from the beginning of the video."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_id = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_id, frame
            frame_id += 1

    def get_frame(self, frame_id: int) -> np.ndarray:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self._cap.read()
        if not ret:
            raise ValueError(f"Cannot read frame {frame_id}")
        return frame

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
