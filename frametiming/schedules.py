"""Frame-timing schedules: WHEN the fake camera sends each frame.

A schedule is a list of times in seconds since the start of the run, one per
frame. For example, [0.0, 0.033, 0.067] means "frame 0 at 0 s, frame 1 at 33 ms,
frame 2 at 67 ms". Every schedule averages roughly `fps` frames per second.
The difference is HOW EVENLY the frames arrive.
"""
from typing import Literal

import numpy as np

ScheduleKind = Literal["steady", "burst", "stall", "jitter"]
SCHEDULES: tuple[str, ...] = ("steady", "burst", "stall", "jitter")

BURST_SIZE = 10            # frames per burst
BURST_SPACING = 0.001      # 1 ms between frames inside a burst
STALL_STARTS = (2.5, 5.0, 7.5)  # seconds at which the camera freezes
STALL_LENGTH = 1.0         # how long each freeze lasts
JITTER_SD = 0.015          # std-dev of random timing noise, in seconds


def make_schedule(
    kind: ScheduleKind, n_frames: int = 300, fps: float = 30.0, seed: int = 0
) -> list[float]:
    """Return the emit time (seconds) of each of `n_frames` frames."""
    steady = [i / fps for i in range(n_frames)]

    if kind == "steady":
        # A perfect camera: one frame every 1/fps seconds.
        return steady

    if kind == "burst":
        # Frames come in clumps of 10, 1 ms apart, one clump every 10/fps s.
        # Same average rate as steady, very uneven arrival.
        period = BURST_SIZE / fps
        return [(i // BURST_SIZE) * period + (i % BURST_SIZE) * BURST_SPACING
                for i in range(n_frames)]

    if kind == "stall":
        # Steady, except the camera freezes for 1 s three times. The frames it
        # "should" have sent during a freeze all arrive at once afterwards,
        # which is what a network hiccup or a stuck encoder looks like.
        out = []
        backlog_position = {s: 0 for s in STALL_STARTS}
        for t in steady:
            for s in STALL_STARTS:
                if s <= t < s + STALL_LENGTH:
                    k = backlog_position[s]
                    backlog_position[s] += 1
                    t = s + STALL_LENGTH + k * BURST_SPACING
                    break
            out.append(t)
        # Keep frames in order: a frame due right as a freeze ends must still
        # come after the backlog that was released at that moment.
        for i in range(1, len(out)):
            out[i] = max(out[i], out[i - 1] + BURST_SPACING)
        return out

    if kind == "jitter":
        # Each gap between frames is 1/fps plus random noise (never negative).
        rng = np.random.default_rng(seed)
        gaps = np.maximum(0.0, 1 / fps + rng.normal(0.0, JITTER_SD, n_frames - 1))
        return [0.0] + np.cumsum(gaps).tolist()

    raise ValueError(f"unknown schedule kind: {kind!r}; expected one of {SCHEDULES}")
