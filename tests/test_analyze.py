"""Tests for turning per-frame logs into summary numbers and charts."""
import numpy as np
import pandas as pd
import pytest

from frametiming.analyze import (
    SLEEP_WORK_MS, check_integrity, column_label, plot_strategy_grid, plot_timeline, summarize)
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


def test_column_label_separates_every_frame_variants():
    plain = summarize(make_run([T], run_id="burst|default|default|every_frame|sleep")).iloc[0]
    wait = summarize(make_run([T], run_id="burst|WAIT|default|every_frame|sleep")).iloc[0]
    assert column_label(plain) != column_label(wait)
    assert "WAIT" in column_label(wait)


# ---------- data integrity checks ----------

def clean_run(mode="sleep"):
    # every other frame processed, 50 ms latency: results 66 ms apart, all >= 40 ms work
    return make_run([T, F] * 5, latencies_ms=[50.0] * 5,
                    run_id=f"steady|WAIT|EAGER|legacy|{mode}")


def test_integrity_clean_run_has_no_problems():
    assert check_integrity(clean_run()) == []


def test_integrity_flags_missing_frame():
    df = clean_run().drop(index=3)
    assert any("missing" in p for p in check_integrity(df))


def test_integrity_flags_frame_never_emitted():
    df = clean_run()
    df.loc[4, "emit_t"] = np.nan
    assert any("never sent" in p for p in check_integrity(df))


def test_integrity_flags_latency_shorter_than_the_work():
    df = make_run([T, F] * 5, latencies_ms=[50, 50, 10, 50, 50])
    assert any("faster than" in p for p in check_integrity(df))


def test_integrity_flags_results_closer_than_the_work():
    df = make_run([T, T, F, F], latencies_ms=[50, 50])  # results 33 ms apart < 40 ms of work
    assert any("apart" in p for p in check_integrity(df))


def test_integrity_skips_work_checks_for_real_model_runs():
    df = make_run([T, T, F, F], latencies_ms=[5, 5], run_id="steady|WAIT|EAGER|legacy|model")
    assert check_integrity(df) == []


def test_sleep_work_matches_runner_default():
    from frametiming.run import RunConfig
    assert SLEEP_WORK_MS == RunConfig.__dataclass_fields__["sleep_ms"].default
