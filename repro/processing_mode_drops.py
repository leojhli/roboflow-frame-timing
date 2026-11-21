"""Minimal reproduction: on a LIVE source, the 1.8 processing modes keep the
source's default ADAPTIVE_DROP_OLDEST buffer, which drops frames before the
mode ever sees them:
  * video_processing_mode="every_frame" still drops frames unless the buffer
    filling strategy is also WAIT;
  * video_processing_mode="auto" processes far fewer frames than the model
    could handle, unless the buffer filling strategy is also WAIT.

Self-contained, needs only:  pip install inference
Run:                         python repro/every_frame_drops.py

Setup: a fake live camera (is_file=False) sends 150 frames at an average of
30 fps, but in bursts of 10 frames 1 ms apart (common for real IP cameras).
Processing each frame takes 40 ms, slightly slower than the camera.
"""
import time
from importlib.metadata import version

import numpy as np
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import SourceProperties, VideoFrameProducer
from inference.core.interfaces.camera.video_source import BufferFillingStrategy

N_FRAMES, FPS, BURST, WORK_S, REPEATS = 150, 30.0, 10, 0.040, 3


class BurstyLiveCamera(VideoFrameProducer):
    def __init__(self):
        self.i, self.start = -1, None
        self.image = np.zeros((360, 640, 3), np.uint8)

    def grab(self) -> bool:
        self.i += 1
        if self.i >= N_FRAMES:
            return False  # end of stream
        due = (self.i // BURST) * BURST / FPS + (self.i % BURST) * 0.001
        self.start = self.start or time.perf_counter()
        while time.perf_counter() < self.start + due:  # block until the frame "arrives"
            time.sleep(0.0005)
        return True

    def retrieve(self):
        return True, self.image

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass

    def discover_source_properties(self) -> SourceProperties:
        return SourceProperties(width=640, height=360, total_frames=-1, is_file=False,
                                fps=FPS, is_reconnectable=False)


def frames_processed(**pipeline_kwargs) -> int:
    processed = 0

    def on_video_frame(frames):
        nonlocal processed
        time.sleep(WORK_S)  # stand-in for a model that takes 40 ms
        processed += len(frames)
        return [None] * len(frames)

    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=BurstyLiveCamera, on_video_frame=on_video_frame, **pipeline_kwargs)
    pipeline.start(use_main_thread=False)
    pipeline.join()
    return processed


if __name__ == "__main__":
    wait = dict(source_buffer_filling_strategy=BufferFillingStrategy.WAIT)
    cases = {
        '"every_frame"': dict(video_processing_mode="every_frame"),
        '"every_frame" + filling=WAIT': dict(video_processing_mode="every_frame", **wait),
        '"auto"': dict(video_processing_mode="auto"),
        '"auto" + filling=WAIT': dict(video_processing_mode="auto", **wait),
    }
    print(f"inference {version('inference')}: {N_FRAMES} frames sent per run, {REPEATS} runs each, "
          f"model capacity ~{100 * (1 / WORK_S) / FPS:.0f}% of frames")
    for label, kwargs in cases.items():
        counts = [frames_processed(**kwargs) for _ in range(REPEATS)]
        print(f"  video_processing_mode={label:32s} processed {counts} of {N_FRAMES}")
