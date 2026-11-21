"""Run one experiment: fake camera -> InferencePipeline -> per-frame log.

For every frame the camera SENDS, we record whether Inference PROCESSED it and,
if so, how long after it was sent the result was ready (latency).

Usage:
    python -m frametiming.run --grid           # full grid, ~10 s per run
    python -m frametiming.run --grid --quick   # 60 frames per run, for testing
"""
import argparse
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)

from frametiming.producer import producer_factory
from frametiming.schedules import SCHEDULES, make_schedule
from frametiming.stamp import read_index

FILLINGS = ("WAIT", "DROP_OLDEST", "ADAPTIVE_DROP_OLDEST", "DROP_LATEST")
CONSUMPTIONS = ("EAGER", "LAZY")
PROCESSING_MODES = ("auto", "every_frame", "freshest")
WARMUP_RUNS = 3  # untimed model inferences before each real-model run
COLUMNS = ["run_id", "schedule", "filling", "consumption", "processing_mode", "mode",
           "frame_idx", "emit_t", "processed", "done_t", "latency_ms"]


@dataclass
class RunConfig:
    schedule: str
    filling: str                    # a BufferFillingStrategy name, or "default"
    consumption: str                # a BufferConsumptionStrategy name, or "default"
    processing_mode: str = "legacy"  # "legacy" or one of PROCESSING_MODES
    mode: Literal["sleep", "model"] = "sleep"
    sleep_ms: float = 40.0           # sleep mode: pretend each frame takes this long
    model_id: str = "yolov8n-640"    # model mode: a real Roboflow model
    n_frames: int = 300
    fps: float = 30.0
    seed: int = 0
    video_path: str | None = None

    @property
    def run_id(self) -> str:
        return f"{self.schedule}|{self.filling}|{self.consumption}|{self.processing_mode}|{self.mode}"


def _make_infer(cfg: RunConfig):
    """Return the function that does the 'work' on a frame."""
    if cfg.mode == "sleep":
        # A stand-in model with a fixed, known cost: results don't depend on
        # this laptop's speed, so they are repeatable.
        return lambda image: time.sleep(cfg.sleep_ms / 1000)

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "model mode needs a Roboflow API key: set the ROBOFLOW_API_KEY "
            "environment variable (never put the key in code or files)")
    from inference import get_model
    model = get_model(model_id=cfg.model_id, api_key=api_key)
    # Warm-up: a model's first few inferences are slow (memory allocation,
    # runtime optimization). Run them now so they don't count as latency.
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    for _ in range(WARMUP_RUNS):
        model.infer(blank)
    return lambda image: model.infer(image)


def run_experiment(cfg: RunConfig) -> pd.DataFrame:
    if cfg.video_path is None:
        from frametiming.video import get_test_video
        cfg.video_path = get_test_video()

    infer = _make_infer(cfg)
    schedule = make_schedule(cfg.schedule, n_frames=cfg.n_frames, fps=cfg.fps, seed=cfg.seed)
    emit_log: dict[int, float] = {}   # frame -> perf_counter() when the camera sent it
    done_log: dict[int, float] = {}   # frame -> perf_counter() when its result was ready
    lock = threading.Lock()

    def on_video_frame(frames: list[VideoFrame]) -> list[None]:
        for frame in frames:
            idx = read_index(frame.image)   # which frame is this? (Task 3 stamp)
            infer(frame.image)
            with lock:
                done_log.setdefault(idx, time.perf_counter())
        return [None] * len(frames)

    def strategy(enum, name):
        return None if name == "default" else enum[name]

    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=producer_factory(cfg.video_path, schedule, emit_log),
        on_video_frame=on_video_frame,
        source_buffer_filling_strategy=strategy(BufferFillingStrategy, cfg.filling),
        source_buffer_consumption_strategy=strategy(BufferConsumptionStrategy, cfg.consumption),
        video_processing_mode=None if cfg.processing_mode == "legacy" else cfg.processing_mode,
    )
    pipeline.start(use_main_thread=False)
    pipeline.join()

    t_ref = emit_log.get(0, min(emit_log.values(), default=0.0))
    rows = []
    for i in range(cfg.n_frames):
        emit = emit_log.get(i, np.nan)
        done = done_log.get(i, np.nan)
        rows.append({
            "run_id": cfg.run_id, "schedule": cfg.schedule, "filling": cfg.filling,
            "consumption": cfg.consumption, "processing_mode": cfg.processing_mode,
            "mode": cfg.mode, "frame_idx": i,
            "emit_t": emit - t_ref, "processed": i in done_log,
            "done_t": done - t_ref, "latency_ms": (done - emit) * 1000,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def grid_configs(n_frames: int = 300, include_model: bool = True,
                 include_sleep: bool = True) -> list[RunConfig]:
    configs = []
    if include_sleep:
        configs += [RunConfig(s, f, c, n_frames=n_frames)
                    for s in SCHEDULES for f in FILLINGS for c in CONSUMPTIONS]
        configs += [RunConfig(s, "default", "default", processing_mode=m, n_frames=n_frames)
                    for s in SCHEDULES for m in PROCESSING_MODES]
        # The processing modes keep the live default ADAPTIVE_DROP_OLDEST buffer,
        # which drops frames before the mode ever sees them. With WAIT filling the
        # mode alone decides what to drop (see docs/api-notes.md).
        configs += [RunConfig(s, "WAIT", "default", processing_mode=m, n_frames=n_frames)
                    for s in SCHEDULES for m in ("every_frame", "auto")]
    if include_model:
        configs += [RunConfig(s, "ADAPTIVE_DROP_OLDEST", "EAGER", mode="model", n_frames=n_frames)
                    for s in SCHEDULES]
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", action="store_true", help="run every combination")
    parser.add_argument("--quick", action="store_true", help="60 frames per run instead of 300")
    parser.add_argument("--only", choices=["sleep", "model"],
                        help="run only the sleep-mode or only the real-model part of the grid")
    parser.add_argument("--out", default="results/runs.csv")
    args = parser.parse_args()
    if not args.grid:
        parser.error("nothing to do: pass --grid")

    logging.getLogger("inference").setLevel(logging.WARNING)
    has_key = bool(os.environ.get("ROBOFLOW_API_KEY"))
    want_model = args.only != "sleep"
    if want_model and not has_key:
        if args.only == "model":
            parser.error("--only model needs the ROBOFLOW_API_KEY environment variable")
        print("ROBOFLOW_API_KEY not set: skipping the real-model runs.")
    configs = grid_configs(n_frames=60 if args.quick else 300,
                           include_model=want_model and has_key,
                           include_sleep=args.only != "model")

    results = []
    for n, cfg in enumerate(configs, 1):
        df = run_experiment(cfg)
        results.append(df)
        print(f"[{n:2d}/{len(configs)}] {cfg.run_id:55s} "
              f"processed {df.processed.mean():6.1%}  "
              f"median latency {df.latency_ms.median():7.1f} ms")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(results).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
