"""A fake live camera that sends real video frames on a timetable we control.

InferencePipeline reads any camera through two calls, repeated forever:
  grab()     -> "is there a next frame?"  (we wait here until it's due)
  retrieve() -> "give me that frame"
We play a video file, but pretend to be a live camera (is_file=False), so
Inference has to cope with frames arriving whether or not it is ready.
"""
import time
from typing import Callable

import cv2
import numpy as np
from inference.core.interfaces.camera.entities import (
    SourceProperties,
    VideoFrameProducer,
)

from frametiming.stamp import stamp_index

MAX_CACHED_FRAMES = 150  # ~5 s of video kept in memory, then it loops
SPIN_MARGIN = 0.002      # last 2 ms of each wait are busy-waited for precision


def _load_frames(video_path: str, frame_width: int) -> list[np.ndarray]:
    """Decode up to MAX_CACHED_FRAMES frames into memory, resized to frame_width.

    Pre-loading matters: decoding a 1080p frame takes several ms, which would
    smear a "10 frames 1 ms apart" burst into 50+ ms. From memory it's instant.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < MAX_CACHED_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        size = (frame_width, round(h * frame_width / w))
        frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise ValueError(f"could not read any frames from {video_path!r}")
    return frames


def _wait_until(deadline: float) -> None:
    """Sleep until perf_counter() reaches deadline. Returns at once if it's past."""
    remaining = deadline - time.perf_counter()
    if remaining > SPIN_MARGIN:
        time.sleep(remaining - SPIN_MARGIN)
    while time.perf_counter() < deadline:
        pass


class ScheduledFrameProducer(VideoFrameProducer):
    def __init__(
        self,
        video_path: str,
        schedule: list[float],
        emit_log: dict[int, float],
        frame_width: int = 640,
    ):
        """
        schedule:  when to send each frame, in seconds after the first grab().
        emit_log:  filled in as we go: frame index -> perf_counter() at send time.
                   Shared with whoever runs the experiment.
        """
        self._frames = _load_frames(video_path, frame_width)
        self._schedule = schedule
        self._emit_log = emit_log
        self._next = 0
        self._start: float | None = None  # clock starts at the first grab()
        self._current: np.ndarray | None = None

    def grab(self) -> bool:
        i = self._next
        if i >= len(self._schedule):
            return False  # end of stream
        # Prepare the frame BEFORE waiting, so sending it costs ~nothing.
        frame = stamp_index(self._frames[i % len(self._frames)], i)
        if self._start is None:
            self._start = time.perf_counter()
        _wait_until(self._start + self._schedule[i])
        self._current = frame
        self._emit_log[i] = time.perf_counter()
        self._next += 1
        return True

    def retrieve(self) -> tuple[bool, np.ndarray]:
        return self._current is not None, self._current

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass

    def discover_source_properties(self) -> SourceProperties:
        h, w = self._frames[0].shape[:2]
        span = self._schedule[-1] if self._schedule else 0.0
        fps = (len(self._schedule) - 1) / span if span > 0 else 30.0
        return SourceProperties(
            width=w, height=h, total_frames=-1, is_file=False, fps=fps,
            is_reconnectable=False,
        )


def producer_factory(
    video_path: str, schedule: list[float], emit_log: dict[int, float], frame_width: int = 640
) -> Callable[[], ScheduledFrameProducer]:
    """InferencePipeline wants a function that BUILDS the camera, not the camera itself."""
    return lambda: ScheduledFrameProducer(video_path, schedule, emit_log, frame_width)
