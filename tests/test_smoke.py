"""Smoke test: can InferencePipeline run on a frame source we wrote ourselves?

If this passes, the whole project idea works: we control exactly when
frames arrive, and Inference treats our source like a live camera.
"""
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)

from tests.helpers import GrayFrameProducer


def test_custom_producer_runs():
    seen: list[int] = []

    def on_video_frame(frames: list[VideoFrame]) -> list[None]:
        seen.extend(f.frame_id for f in frames)
        return [None] * len(frames)

    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=lambda: GrayFrameProducer(n_frames=10),
        on_video_frame=on_video_frame,
        # WAIT + LAZY = never drop a frame, so we must see all 10.
        source_buffer_filling_strategy=BufferFillingStrategy.WAIT,
        source_buffer_consumption_strategy=BufferConsumptionStrategy.LAZY,
    )
    pipeline.start(use_main_thread=False)
    pipeline.join()

    assert len(seen) == 10
