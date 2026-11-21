"""Tests for the experiment runner: fake camera -> InferencePipeline -> log."""
import cv2
import numpy as np
import pytest

from frametiming.run import RunConfig, grid_configs, run_experiment


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("video") / "tiny.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for shade in (50, 100, 150):
        writer.write(np.full((240, 320, 3), shade, np.uint8))
    writer.release()
    return path


def quick(tiny_video, **overrides) -> RunConfig:
    base = dict(schedule="steady", filling="WAIT", consumption="LAZY",
                n_frames=30, fps=100.0, video_path=tiny_video)
    return RunConfig(**{**base, **overrides})


@pytest.fixture(scope="module")
def overloaded_run(tiny_video):
    # Model "takes" 100 ms per frame, camera sends every 10 ms: must drop.
    return run_experiment(quick(tiny_video, filling="DROP_OLDEST",
                                consumption="EAGER", sleep_ms=100))


def test_wait_processes_everything(tiny_video):
    df = run_experiment(quick(tiny_video, sleep_ms=0))
    assert df.processed.all()


def test_overload_drops_frames(overloaded_run):
    assert 0 < overloaded_run.processed.sum() < 30


def test_latency_non_negative(overloaded_run):
    assert (overloaded_run.latency_ms.dropna() >= 0).all()


def test_one_row_per_scheduled_frame(overloaded_run):
    assert sorted(overloaded_run.frame_idx) == list(range(30))


def test_dropped_frames_have_no_latency(overloaded_run):
    dropped = overloaded_run[~overloaded_run.processed]
    assert dropped.latency_ms.isna().all() and dropped.done_t.isna().all()


def test_columns_and_run_id(overloaded_run):
    assert list(overloaded_run.columns) == [
        "run_id", "schedule", "filling", "consumption", "processing_mode", "mode",
        "frame_idx", "emit_t", "processed", "done_t", "latency_ms"]
    assert overloaded_run.run_id.iloc[0] == "steady|DROP_OLDEST|EAGER|legacy|sleep"
    assert overloaded_run.emit_t.iloc[0] == 0.0


@pytest.mark.parametrize("processing_mode", ["auto", "every_frame", "freshest"])
def test_processing_modes_run(tiny_video, processing_mode):
    df = run_experiment(quick(tiny_video, filling="default", consumption="default",
                              processing_mode=processing_mode, sleep_ms=20))
    assert len(df) == 30 and df.processed.sum() > 0


@pytest.mark.timeout(30)
def test_run_terminates(tiny_video):
    run_experiment(quick(tiny_video, schedule="burst", filling="DROP_LATEST",
                         consumption="EAGER", sleep_ms=50))


def test_model_mode_requires_api_key(tiny_video, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ROBOFLOW_API_KEY"):
        run_experiment(quick(tiny_video, mode="model"))


def test_grid_has_every_combination():
    ids = {c.run_id for c in grid_configs(n_frames=60, include_model=True)}
    assert len(ids) == 4 * 4 * 2 + 4 * 3 + 4  # legacy grid + processing modes + model
