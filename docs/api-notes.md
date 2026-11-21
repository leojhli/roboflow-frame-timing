# Inference API notes (verified against `inference==1.8.5`)

Checked by reading the installed source and by `tests/test_smoke.py` passing.

## Where things live

In 1.8.x, the video code moved into a separate package called `streamvision`. The old import paths still work through compatibility shims, and we use the old paths because that's what Roboflow's docs show.

| Name | Import |
|---|---|
| `InferencePipeline` | `from inference import InferencePipeline` |
| `VideoFrameProducer`, `SourceProperties`, `VideoFrame` | `from inference.core.interfaces.camera.entities import ...` |
| `BufferFillingStrategy`, `BufferConsumptionStrategy` | `from inference.core.interfaces.camera.video_source import ...` |

Real source: `streamvision/camera/entities.py`, `streamvision/camera/video_source.py`, `streamvision/stream/pipeline.py`.

## Custom frame source: supported ✅

`VideoSourceIdentifier = Union[str, int, Callable[[], VideoFrameProducer]]`, so we pass a **function that builds our producer** as `video_reference`. No RTSP/MJPEG fallback is needed.

`VideoFrameProducer` methods: `grab() -> bool`, `retrieve() -> (bool, image)`, `release()`, `isOpened() -> bool`, `discover_source_properties() -> SourceProperties`. Optional: `initialize_source_properties(dict)` and `connection_error_message() -> str`.

`SourceProperties` fields: `width, height, total_frames, is_file, fps, is_reconnectable=None, timestamp_created=None`.

**`is_file=False` is what makes Inference treat the source as a live camera.** The live and file paths behave differently:

| | File (`is_file=True`) | Live (`is_file=False`) |
|---|---|---|
| Default filling strategy | `WAIT` (never drop) | `ADAPTIVE_DROP_OLDEST` |
| Default consumption strategy | `LAZY` | `EAGER` |

## Running custom logic

```python
pipeline = InferencePipeline.init_with_custom_logic(
    video_reference=lambda: MyProducer(...),
    on_video_frame=fn,                      # receives list[VideoFrame], returns list of predictions
    source_buffer_filling_strategy=BufferFillingStrategy.X,
    source_buffer_consumption_strategy=BufferConsumptionStrategy.Y,
)
pipeline.start(use_main_thread=False)
pipeline.join()
```

- `on_video_frame` gets a **list** of `VideoFrame`s (one per source; we have one). It must return a list of the same length.
- `VideoFrame` has `image`, `frame_id`, `frame_timestamp`, `source_id`, `fps`, `measured_fps`, `comes_from_video_file`, `total_frames`.
- Filling strategies: `WAIT, DROP_OLDEST, ADAPTIVE_DROP_OLDEST, DROP_LATEST, ADAPTIVE_DROP_LATEST`. Consumption: `LAZY, EAGER`.

## New in 1.8.x, not in the original plan: processing modes

`init` / `init_with_custom_logic` also take `video_processing_mode` and `max_staleness`:

- `None` (default): the legacy behavior, controlled only by the two buffer strategies above.
- `"auto"`: serves every frame in order while the consumer keeps up. When overloaded, it drops frames older than `max_staleness` (default **0.5 s**) and reports the drops.
- `"every_frame"`: strict in-order processing with no drops. Under overload, latency grows until the buffer is full.
- `"freshest"`: always takes the newest frame and silently skips the rest. Lowest latency.

Roboflow wrote these to fix bursty real cameras (from the module docstring: cameras "emit frames in bursts separated by encoder pauses of 400-500 ms"). That is almost exactly our "burst" and "stall" schedules, so comparing these modes is likely the most useful result for the Inference team.

## Practical notes

- `import inference` takes about 20–30 s on this machine, so each pytest run has a slow start.
- `opencv-python`, `opencv-contrib-python` and `opencv-python-headless` are all installed by dependencies. `cv2` resolves to 4.11.0, which works.

## Observed: the processing modes keep the source's dropping buffer

`video_processing_mode` only controls the *collection* step, i.e. how frames are taken from the source buffer. The source's own decoding buffer still uses the live-stream default filling strategy, `ADAPTIVE_DROP_OLDEST`, which drops frames before the mode ever sees them. Full 300-frame runs, 40 ms of work per frame, camera averaging 30 fps (ceiling: 83% of frames):

| Schedule | `auto` | `auto` + `WAIT` | `every_frame` | `every_frame` + `WAIT` |
|---|---|---|---|---|
| burst | 29% (p95 354 ms) | **79%** (p95 523 ms) | 30% (p95 382 ms) | **100%** (p95 2,807 ms) |
| jitter | 47% (p95 414 ms) | **80%** (p95 516 ms) | 48% (p95 535 ms) | **100%** (p95 2,847 ms) |

- To really get every frame from a live source, pass **both** `video_processing_mode="every_frame"` and `source_buffer_filling_strategy=BufferFillingStrategy.WAIT`. The cost is latency that grows while the model is slower than the camera.
- `auto` + `WAIT` gets close to the model's capacity while keeping p95 near `auto`'s 0.5 s staleness budget.
- Verified on 1.8.5 and 1.8.6 with `repro/processing_mode_drops.py`.
