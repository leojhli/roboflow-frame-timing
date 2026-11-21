"""Tests for the scheduled fake camera."""
import time

import cv2
import numpy as np
import pytest

from frametiming.producer import ScheduledFrameProducer, producer_factory
from frametiming.stamp import read_index


@pytest.fixture
def tiny_video(tmp_path) -> str:
    """A 3-frame 320x240 video file, so tests don't need a download."""
    path = str(tmp_path / "tiny.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for shade in (50, 100, 150):
        writer.write(np.full((240, 320, 3), shade, np.uint8))
    writer.release()
    return path


def emit_all(producer: ScheduledFrameProducer) -> list[int]:
    indices = []
    while producer.grab():
        ok, frame = producer.retrieve()
        assert ok
        indices.append(read_index(frame))
    return indices


def test_emits_all_scheduled_frames_then_stops(tiny_video):
    # 5 frames scheduled but the video only has 3, so it must loop.
    p = ScheduledFrameProducer(tiny_video, [0, 0.01, 0.02, 0.03, 0.04], emit_log={})
    assert emit_all(p) == [0, 1, 2, 3, 4]
    assert p.grab() is False  # stays finished


def test_emit_times_follow_schedule(tiny_video):
    schedule = [0, 0.05, 0.10]
    log: dict[int, float] = {}
    emit_all(ScheduledFrameProducer(tiny_video, schedule, emit_log=log))
    for i, planned in enumerate(schedule):
        actual = log[i] - log[0]  # log holds absolute perf_counter() times
        assert planned - 0.005 <= actual < planned + 0.03


def test_schedule_starts_at_first_grab(tiny_video):
    # The pipeline takes a while to start. That delay must NOT make the
    # first frames "late" and come out bunched together.
    log: dict[int, float] = {}
    p = ScheduledFrameProducer(tiny_video, [0, 0.05, 0.10], emit_log=log)
    time.sleep(0.2)
    emit_all(p)
    assert log[2] - log[0] >= 0.095


def test_zero_interval_schedule_emits_without_error(tiny_video):
    p = ScheduledFrameProducer(tiny_video, [0, 0, 0, 0], emit_log={})
    start = time.perf_counter()
    assert emit_all(p) == [0, 1, 2, 3]
    assert time.perf_counter() - start < 0.05


def test_frames_resized_to_frame_width(tiny_video):
    p = ScheduledFrameProducer(tiny_video, [0], emit_log={}, frame_width=640)
    p.grab()
    assert p.retrieve()[1].shape == (480, 640, 3)


def test_reports_live_source(tiny_video):
    props = ScheduledFrameProducer(tiny_video, [0, 0.1], emit_log={}).discover_source_properties()
    assert props.is_file is False
    assert (props.width, props.height) == (640, 480)


def test_factory_builds_fresh_producer(tiny_video):
    make = producer_factory(tiny_video, [0, 0.01], emit_log={})
    assert isinstance(make(), ScheduledFrameProducer)
