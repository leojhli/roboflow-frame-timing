"""Tests for turning per-frame logs into summary numbers and charts."""
import numpy as np
import pandas as pd
import pytest

from frametiming.analyze import plot_strategy_grid, plot_timeline, summarize
from frametiming.run import COLUMNS

T, F = True, False


def make_run(processed, latencies_ms=None, done_order=None, run_id="steady|WAIT|EAGER|legacy|sleep"):
    """Build a fake run_experiment() result by hand."""
    n = len(processed)
    schedule, filling, consumption, processing_mode, mode = run_id.split("|")
    emit = np.arange(n) / 30
    lat = np.full(n, np.nan)
    lat[np.array(processed)] = latencies_ms if latencies_ms is not None else 50.0
    done = emit + lat / 1000
    if done_order is not None:  # force a specific finishing order
        for rank, idx in enumerate(done_order):
            done[idx] = 10 + rank
    return pd.DataFrame({
        "run_id": run_id, "schedule": schedule, "filling": filling,
        "consumption": consumption, "processing_mode": processing_mode, "mode": mode,
        "frame_idx": range(n), "emit_t": emit, "processed": processed,
        "done_t": done, "latency_ms": lat,
    }, columns=COLUMNS)


def test_summarize_basic():
    df = make_run([T, T, F, F, F, T, T, F, T, T], latencies_ms=[10, 20, 30, 40, 50, 60])
    row = summarize(df).iloc[0]
    assert row.delivered_pct == 60.0
    assert row.max_drop_run == 3
    assert row.p50_latency_ms == 35.0
    assert row.p95_latency_ms == pytest.approx(57.5)


def test_summarize_all_dropped():
    row = summarize(make_run([F] * 8)).iloc[0]
    assert row.delivered_pct == 0.0
    assert np.isnan(row.p50_latency_ms) and np.isnan(row.p95_latency_ms)
    assert row.max_drop_run == 8


def test_out_of_order_counted():
    df = make_run([T, T, T], done_order=[0, 2, 1])
    assert summarize(df).iloc[0].out_of_order == 1


def test_summarize_one_row_per_run_with_labels():
    df = pd.concat([make_run([T, F]), make_run([T, T], run_id="burst|default|default|auto|sleep")])
    s = summarize(df)
    assert len(s) == 2
    assert {"schedule", "filling", "consumption", "processing_mode", "mode"} <= set(s.columns)


def test_plots_handle_all_dropped(tmp_path):
    df = make_run([F] * 8)
    plot_timeline(df, tmp_path / "timeline.png")
    plot_strategy_grid(summarize(df), tmp_path / "grid.png")
    assert (tmp_path / "timeline.png").stat().st_size > 0
    assert (tmp_path / "grid.png").stat().st_size > 0
