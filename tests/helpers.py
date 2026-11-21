"""Test-only helpers."""
import numpy as np
from inference.core.interfaces.camera.entities import (
    SourceProperties,
    VideoFrameProducer,
)


class GrayFrameProducer(VideoFrameProducer):
    """A fake 'camera' that instantly emits `n_frames` plain gray images.

    InferencePipeline talks to any frame source through the same small set of
    methods OpenCV's VideoCapture uses: grab() = "is there a next frame?",
    retrieve() = "give me that frame", plus a few for setup and cleanup.
    """

    def __init__(self, n_frames: int, width: int = 64, height: int = 64):
        self._n_frames = n_frames
        self._emitted = 0
        self._image = np.full((height, width, 3), 128, dtype=np.uint8)

    def isOpened(self) -> bool:
        return True

    def grab(self) -> bool:
        if self._emitted >= self._n_frames:
            return False  # end of stream
        self._emitted += 1
        return True

    def retrieve(self) -> tuple[bool, np.ndarray]:
        return True, self._image.copy()

    def release(self) -> None:
        pass

    def discover_source_properties(self) -> SourceProperties:
        h, w = self._image.shape[:2]
        # is_file=False is the key line: Inference treats us as a LIVE camera,
        # so its frame-dropping buffer logic is switched on.
        return SourceProperties(
            width=w, height=h, total_frames=-1, is_file=False, fps=30.0,
            is_reconnectable=False,
        )
