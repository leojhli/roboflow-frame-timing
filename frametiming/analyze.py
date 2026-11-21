"""Turn per-frame logs into summary numbers and charts.

Usage:
    python -m frametiming.analyze results/runs.csv
writes results/summary.csv and results/plots/*.png
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SCHEDULE_ORDER = ["steady", "burst", "stall", "jitter"]
FILLING_ORDER = ["WAIT", "DROP_OLDEST", "ADAPTIVE_DROP_OLDEST", "DROP_LATEST"]
LABEL_COLS = ["schedule", "filling", "consumption", "processing_mode", "mode"]

# Colors: validated colorblind-safe pair + one-hue blue ramp for magnitudes.
PROCESSED = "#2a78d6"
DROPPED = "#eb6834"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10,
})


# ---------- numbers ----------

def _longest_run(flags: np.ndarray) -> int:
    """Length of the longest streak of True values."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _out_of_order(run: pd.DataFrame) -> int:
    """How many results came back with an OLDER frame than the result before them."""
    done = run[run.processed].sort_values("done_t").frame_idx.to_numpy()
    return int((np.diff(done) < 0).sum())


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run: how much got through, and how late."""
    rows = []
    for run_id, run in df.groupby("run_id", sort=False):
        run = run.sort_values("frame_idx")
        lat = run.latency_ms.dropna()
        rows.append({
            "run_id": run_id,
            **{c: run[c].iloc[0] for c in LABEL_COLS},
            "frames": len(run),
            "delivered_pct": 100.0 * run.processed.mean(),
            "p50_latency_ms": lat.median() if len(lat) else np.nan,
            "p95_latency_ms": lat.quantile(0.95) if len(lat) else np.nan,
            "max_drop_run": _longest_run(~run.processed.to_numpy(dtype=bool)),
            "out_of_order": _out_of_order(run),
        })
    return pd.DataFrame(rows)


# ---------- integrity checks ----------

SLEEP_WORK_MS = 40.0  # must match RunConfig.sleep_ms (a test checks this)
CLOCK_SLACK_MS = 0.5
# Longest a result may arrive after the worker was free and its frame existed.
# Normal runs: <= ~180 ms (pipeline collection windows). Real models get more
# room because their speed varies. Anything above means the process froze.
MAX_UNEXPLAINED_WAIT_MS = {"sleep": 1_000.0, "model": 5_000.0}


def _longest_unexplained_wait_ms(done: pd.DataFrame, work_ms: float) -> float:
    """Longest delay not explained by work: each result can start once the worker
    is free AND its frame has arrived, then needs `work_ms`."""
    d = done.sort_values("done_t")
    t_done, t_emit = d.done_t.to_numpy(), d.emit_t.to_numpy()
    if len(t_done) < 2:
        return 0.0
    waits = (t_done[1:] - np.maximum(t_done[:-1], t_emit[1:])) * 1000 - work_ms
    return float(waits.max())


def check_integrity(df: pd.DataFrame, sleep_work_ms: float = SLEEP_WORK_MS) -> list[str]:
    """Physical sanity checks on the measurements. Returns problems found ([] = clean).

    Sleep-mode runs do exactly `sleep_work_ms` of work per frame on one worker,
    so no result can arrive sooner than that after its frame was sent, and no
    two results can finish closer together than that. Breaking either rule
    would mean the measurement itself is wrong.
    """
    problems = []
    for run_id, run in df.groupby("run_id", sort=False):
        idx = run.frame_idx.to_numpy()
        missing = sorted(set(range(idx.max() + 1)) - set(idx)) if len(idx) else []
        if missing or len(idx) != len(set(idx)):
            problems.append(f"{run_id}: frames missing or duplicated ({missing[:5]}...)")
        if run.emit_t.isna().any():
            problems.append(f"{run_id}: {int(run.emit_t.isna().sum())} frames never sent by the camera")
        done = run[run.processed]
        if (done.latency_ms < 0).any():
            problems.append(f"{run_id}: negative latency")
        mode = run["mode"].iloc[0]
        pause_ms = _longest_unexplained_wait_ms(done, SLEEP_WORK_MS if mode == "sleep" else 0.0)
        if pause_ms > MAX_UNEXPLAINED_WAIT_MS.get(mode, 5_000.0):
            problems.append(f"{run_id}: the process paused for {pause_ms / 1000:.1f} s mid-run "
                            "(did the computer sleep?) - re-run it")
        if mode != "sleep" or done.empty:
            continue
        floor = sleep_work_ms - CLOCK_SLACK_MS
        if done.latency_ms.min() < floor:
            problems.append(f"{run_id}: a result came back faster than the "
                            f"{sleep_work_ms:.0f} ms of work ({done.latency_ms.min():.1f} ms)")
        gaps_ms = np.diff(np.sort(done.done_t.to_numpy())) * 1000
        if len(gaps_ms) and gaps_ms.min() < floor:
            problems.append(f"{run_id}: two results finished only {gaps_ms.min():.1f} ms apart "
                            f"(one worker needs {sleep_work_ms:.0f} ms each)")
    return problems


# ---------- charts ----------

def describe_run(run_id: str) -> str:
    """Human-readable chart title for one run, naming every setting that differs."""
    schedule, filling, consumption, processing_mode, mode = run_id.split("|")
    if processing_mode != "legacy":
        how = f"video_processing_mode={processing_mode}"
        how += "" if filling == "default" else f" + filling={filling}"
    else:
        how = f"filling={filling}, consumption={consumption}"
    work = "YOLOv8n model" if mode == "model" else "fixed 40 ms work per frame"
    return f"{schedule} camera · {how} · {work}"


def plot_timeline(run: pd.DataFrame, path) -> None:
    """Top: which frames got through. Bottom: how late each result was."""
    run = run.sort_values("frame_idx")
    ok, lost = run[run.processed], run[~run.processed]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 2.4]})

    top.scatter(ok.emit_t, np.ones(len(ok)), marker="|", s=220, linewidths=2,
                color=PROCESSED, label=f"processed ({len(ok)})")
    top.scatter(lost.emit_t, np.zeros(len(lost)), marker="|", s=220, linewidths=2,
                color=DROPPED, label=f"dropped ({len(lost)})")
    top.set_yticks([1, 0], ["processed", "dropped"])
    top.set_ylim(-0.6, 1.6)
    top.spines["left"].set_visible(False)
    top.tick_params(axis="y", length=0)
    top.legend(loc="upper right", bbox_to_anchor=(1, 1.45), ncol=2, frameon=False)

    if len(ok):
        # Dots only: a connecting line would invent results across dropped gaps.
        bottom.scatter(ok.emit_t, ok.latency_ms, s=30, color=PROCESSED,
                       edgecolors=SURFACE, linewidths=1, zorder=3)
    else:
        bottom.text(0.5, 0.5, "no frames processed", transform=bottom.transAxes,
                    ha="center", va="center", color=MUTED)
    bottom.set_ylim(bottom=0)
    bottom.set_ylabel("latency (ms)")
    bottom.set_xlabel("time the camera sent the frame (s)")
    bottom.grid(axis="y", color=GRID, linewidth=0.8)
    bottom.set_axisbelow(True)

    pct = 100 * run.processed.mean()
    fig.suptitle(describe_run(run.run_id.iloc[0]), x=0.06, ha="left", fontsize=11, color=INK)
    fig.text(0.06, 0.905, f"{pct:.0f}% of frames processed", color=INK_2, fontsize=9)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.12, hspace=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


FILLING_LABEL = {"WAIT": "WAIT", "DROP_OLDEST": "DROP_\nOLDEST",
                 "ADAPTIVE_DROP_OLDEST": "ADAPTIVE_\nDROP_OLDEST", "DROP_LATEST": "DROP_\nLATEST"}
MODE_ORDER = ["auto", "every_frame", "freshest"]


def column_label(row) -> str:
    """Short heatmap column name for one run's settings."""
    if row.processing_mode != "legacy":
        extra = "" if row.filling == "default" else f"\n+ {row.filling}"
        return f"mode=\n{row.processing_mode}{extra}"
    return f"{FILLING_LABEL.get(row.filling, row.filling)}\n+ {row.consumption}"


def _order(items: list, item) -> int:
    return items.index(item) if item in items else len(items)


def _column_key(row) -> tuple:
    """Sort columns: legacy settings first (by filling, EAGER before LAZY), then modes."""
    if row.processing_mode != "legacy":
        return (1, _order(MODE_ORDER, row.processing_mode), 0 if row.filling == "default" else 1)
    return (0, _order(FILLING_ORDER, row.filling), _order(["EAGER", "LAZY"], row.consumption))


METRIC_TITLES = {
    "delivered_pct": ("% of frames processed", "{:.0f}%"),
    "p50_latency_ms": ("median latency (ms)", "{:.0f}"),
    "p95_latency_ms": ("95th-percentile latency (ms)", "{:.0f}"),
    "max_drop_run": ("longest streak of dropped frames", "{:.0f}"),
}


def plot_strategy_grid(summary: pd.DataFrame, path, metric: str = "delivered_pct") -> None:
    """Heatmap: camera schedule (rows) x buffer settings (columns)."""
    s = summary[summary["mode"] == "sleep"] if (summary["mode"] == "sleep").any() else summary
    s = s.assign(col=[column_label(r) for r in s.itertuples()],
                 key=[_column_key(r) for r in s.itertuples()])
    cols = list(dict.fromkeys(s.sort_values("key").col))
    rows = [r for r in SCHEDULE_ORDER if r in set(s.schedule)] or sorted(set(s.schedule))
    grid = s.pivot_table(index="schedule", columns="col", values=metric, aggfunc="first",
                         dropna=False).reindex(index=rows, columns=cols)
    values = grid.to_numpy(dtype=float)

    title, fmt = METRIC_TITLES.get(metric, (metric, "{:.1f}"))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    cmap.set_bad(SURFACE)
    finite = values[np.isfinite(values)]
    vmin, vmax = (0, 100) if metric == "delivered_pct" else (
        (0, float(finite.max())) if finite.size else (0, 1))

    fig, ax = plt.subplots(figsize=(max(5, 1.05 * len(cols) + 1.5), 0.62 * len(rows) + 2.2))
    ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax,
                  edgecolors=SURFACE, linewidth=2)
    for (i, j), v in np.ndenumerate(values):
        dark = np.isfinite(v) and (v - vmin) / ((vmax - vmin) or 1) > 0.5
        ax.text(j + 0.5, i + 0.5, "–" if not np.isfinite(v) else fmt.format(v),
                ha="center", va="center", fontsize=9, color="#ffffff" if dark else INK)
    ax.set_xticks(np.arange(len(cols)) + 0.5, cols, fontsize=7.5)
    ax.set_yticks(np.arange(len(rows)) + 0.5, [f"{r} camera" for r in rows])
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- command line ----------

def _slug(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", run_id).strip("_").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_csv", nargs="*", default=["results/runs.csv"],
                        help="one or more CSVs from frametiming.run (they are combined)")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.runs_csv], ignore_index=True)
    out = Path(args.out_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    problems = check_integrity(df)
    print("data checks: all passed" if not problems else
          "DATA CHECKS FAILED:\n  " + "\n  ".join(problems))

    summary = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    for metric in ("delivered_pct", "p95_latency_ms"):
        plot_strategy_grid(summary, plots / f"grid_{metric}.png", metric=metric)
    for run_id, run in df.groupby("run_id", sort=False):
        plot_timeline(run, plots / f"timeline_{_slug(run_id)}.png")
    print(f"wrote {out / 'summary.csv'} and {len(list(plots.glob('*.png')))} plots in {plots}")


if __name__ == "__main__":
    main()
