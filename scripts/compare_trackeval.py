#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DEFAULT_SCORE_METRICS = [
    "HOTA(0)", "LocA(0)", "HOTA___AUC", "DetA___AUC", "AssA___AUC", "MOTA", "IDF1", "CLR_F1"
]
DEFAULT_COUNT_METRICS = ["IDSW", "Frag"]


def find_csvs(results_dir: Path) -> List[Path]:
    return sorted([p for p in results_dir.rglob("*.csv") if p.is_file()])


def infer_fps(text: str) -> Optional[int]:
    """
    Robust fps inference for strings like:
      - pig30fps_vs_...
      - pig_5fps_...
      - fps30 / fps_30
    IMPORTANT: do NOT require word-boundary after 'fps' (underscore is a word char).
    """
    t = str(text).lower()

    # 30fps, 30_fps, 30-fps, 30 fps (no \b!)
    m = re.search(r"(\d+)\s*[_ -]?\s*fps", t)
    if m:
        return int(m.group(1))

    # fps30, fps_30, fps-30, fps 30
    m = re.search(r"fps\s*[_ -]?\s*(\d+)", t)
    if m:
        return int(m.group(1))

    return None


def read_csv_any_shape(p: Path) -> pd.DataFrame:
    """
    Supports:
      A) long format: has 'tracker' col already
      B) wide compare format: first col = metric name, other cols = tracker names
    Returns long-ish: one row per tracker, columns=metrics.
    """
    try:
        raw = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if "tracker" in raw.columns:
        df = raw.copy()
        df["__source_csv__"] = str(p)
        return df

    # Wide compare: first column is metric names
    if raw.shape[1] >= 2:
        metric_col = raw.columns[0]
        long = raw.melt(id_vars=[metric_col], var_name="tracker", value_name="value")
        long = long.rename(columns={metric_col: "metric"})
        df = long.pivot_table(index="tracker", columns="metric", values="value", aggfunc="first").reset_index()
        df.columns.name = None
        df["__source_csv__"] = str(p)
        return df

    return pd.DataFrame()


def load_all_rows(csv_paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        df = read_csv_any_shape(p)
        if not df.empty and "tracker" in df.columns:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_fps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds df['fps'] by:
      1) tracker string
      2) source csv path
      3) per-file fallback: infer from filename and assign to all rows in that CSV
    """
    if df.empty:
        return df

    df = df.copy()

    fps_from_tracker = df["tracker"].astype(str).apply(infer_fps)
    fps_from_path = df["__source_csv__"].astype(str).apply(infer_fps)

    # Use pandas NA so fill logic works reliably
    fps_from_tracker = fps_from_tracker.astype("Float64")
    fps_from_path = fps_from_path.astype("Float64")

    df["fps"] = fps_from_tracker.fillna(fps_from_path)

    # Per-file fallback
    for src, idx in df.groupby("__source_csv__").groups.items():
        if df.loc[idx, "fps"].notna().any():
            continue
        f = infer_fps(src)  # now works for ...30fps_vs...
        if f is not None:
            df.loc[idx, "fps"] = float(f)

    df["fps"] = df["fps"].astype("Int64")  # nullable int
    return df


def mean_metrics(df: pd.DataFrame, metrics: List[str]) -> pd.Series:
    existing = [m for m in metrics if m in df.columns]
    if df.empty or not existing:
        return pd.Series(dtype=float)
    tmp = df[existing].apply(pd.to_numeric, errors="coerce")
    return tmp.mean(axis=0, skipna=True)


def mean_curve(df: pd.DataFrame, prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    if df.empty:
        return np.array([]), np.array([])
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix + "___")]
    pairs = []
    for c in cols:
        suf = c.split("___", 1)[1]
        if suf.isdigit():
            pairs.append((int(suf), c))
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        return np.array([]), np.array([])
    alphas = np.array([k for k, _ in pairs], dtype=int)
    vals = df[[c for _, c in pairs]].apply(pd.to_numeric, errors="coerce").mean(axis=0).to_numpy()
    return alphas, vals


def normalize_scores(vals: np.ndarray) -> np.ndarray:
    """Auto-convert 0..100 style to 0..1 for plotting."""
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return vals
    if np.nanmedian(finite) > 1.5 or np.nanmax(finite) > 2.0:
        return vals / 100.0
    return vals


def plot_bar_scores(fps: int, my_vals: pd.Series, bt_vals: pd.Series, metrics: List[str], out_dir: Path, title: str):
    metrics = [m for m in metrics if (m in my_vals.index) or (m in bt_vals.index)]
    if not metrics:
        return
    my = np.array([my_vals.get(m, np.nan) for m in metrics], dtype=float)
    bt = np.array([bt_vals.get(m, np.nan) for m in metrics], dtype=float)
    my = normalize_scores(my)
    bt = normalize_scores(bt)

    x = np.arange(len(metrics))
    w = 0.38
    plt.figure(figsize=(max(9, len(metrics) * 1.2), 5))
    plt.bar(x - w/2, bt, width=w, label="ByteTrack")
    plt.bar(x + w/2, my, width=w, label="My algorithm")
    plt.xticks(x, metrics, rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title(f"{title} — {fps} fps (score metrics)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.tight_layout()
    plt.savefig(out_dir / f"bar_scores_{fps}fps.png", dpi=200)
    plt.close()


def plot_bar_counts(fps: int, my_vals: pd.Series, bt_vals: pd.Series, metrics: List[str], out_dir: Path, title: str):
    metrics = [m for m in metrics if (m in my_vals.index) or (m in bt_vals.index)]
    if not metrics:
        return
    my = np.array([my_vals.get(m, np.nan) for m in metrics], dtype=float)
    bt = np.array([bt_vals.get(m, np.nan) for m in metrics], dtype=float)

    x = np.arange(len(metrics))
    w = 0.38
    plt.figure(figsize=(max(7, len(metrics) * 1.6), 5))
    plt.bar(x - w/2, bt, width=w, label="ByteTrack")
    plt.bar(x + w/2, my, width=w, label="My algorithm")
    plt.xticks(x, metrics, rotation=35, ha="right")
    plt.ylabel("Count")
    plt.title(f"{title} — {fps} fps (count/other metrics)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"bar_counts_{fps}fps.png", dpi=200)
    plt.close()


def plot_hota_curve(fps: int, my_a: np.ndarray, my_c: np.ndarray, bt_a: np.ndarray, bt_c: np.ndarray,
                    out_dir: Path, title: str):
    plt.figure(figsize=(10, 5))
    if bt_a.size:
        plt.plot(bt_a, bt_c, marker="o", label="ByteTrack")
    if my_a.size:
        plt.plot(my_a, my_c, marker="o", label="My algorithm")
    plt.xlabel("Alpha threshold (e.g., 5 = 0.05)")
    plt.ylabel("HOTA")
    plt.title(f"{title} — HOTA vs alpha — {fps} fps (mean across matched rows)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"hota_curve_{fps}fps.png", dpi=200)
    plt.close()


def choose_best_my_tracker(df_fps: pd.DataFrame, bt_pattern: str, rank_metric: str) -> Optional[str]:
    """
    In your 30fps CSV you have MANY candidates. If user doesn't provide my_pattern,
    we pick the best non-ByteTrack tracker by rank_metric.
    """
    if df_fps.empty:
        return None
    # exclude bytetrack-like trackers
    mask_bt = df_fps["tracker"].astype(str).str.lower().str.contains(bt_pattern.lower(), na=False)
    candidates = df_fps.loc[~mask_bt].copy()
    if candidates.empty:
        return None

    metric = rank_metric if rank_metric in candidates.columns else ("HOTA(0)" if "HOTA(0)" in candidates.columns else None)
    if metric is None:
        return candidates["tracker"].astype(str).iloc[0]

    vals = pd.to_numeric(candidates[metric], errors="coerce")
    best_idx = vals.idxmax()
    if pd.isna(best_idx):
        return candidates["tracker"].astype(str).iloc[0]
    return candidates.loc[best_idx, "tracker"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--bt_pattern", type=str, default="bytetrack")
    ap.add_argument("--my_pattern", type=str, default=None,
                    help="Optional substring to select your algorithm. If omitted, auto-pick best non-ByteTrack per fps.")
    ap.add_argument("--fps", type=int, nargs="+", default=[5, 30])
    ap.add_argument("--title", type=str, default="TrackEval comparison")
    ap.add_argument("--score_metrics", type=str, nargs="+", default=DEFAULT_SCORE_METRICS)
    ap.add_argument("--count_metrics", type=str, nargs="+", default=DEFAULT_COUNT_METRICS)
    ap.add_argument("--rank_metric", type=str, default="HOTA___AUC",
                    help="Metric used to auto-pick best 'my algorithm' when --my_pattern is not given.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "_compare_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = find_csvs(results_dir)
    all_rows = load_all_rows(csvs)
    if all_rows.empty:
        raise SystemExit("No readable CSVs found.")

    all_rows = attach_fps(all_rows)

    if args.debug:
        print("\n[DEBUG] Per-file trackers & inferred fps:")
        for src, g in all_rows.groupby("__source_csv__"):
            trackers = sorted(g["tracker"].astype(str).unique())
            fps_counts = g["fps"].value_counts(dropna=False).to_dict()
            print(f" - {Path(src).name}: trackers={trackers} fps_counts={fps_counts}")

    for fps in args.fps:
        df_f = all_rows[all_rows["fps"] == fps].copy()

        # ByteTrack rows
        bt = df_f[df_f["tracker"].astype(str).str.lower().str.contains(args.bt_pattern.lower(), na=False)].copy()

        # My algorithm rows
        if args.my_pattern:
            my = df_f[df_f["tracker"].astype(str).str.lower().str.contains(args.my_pattern.lower(), na=False)].copy()
            chosen = args.my_pattern
        else:
            best_tracker = choose_best_my_tracker(df_f, args.bt_pattern, args.rank_metric)
            chosen = best_tracker
            my = df_f[df_f["tracker"].astype(str) == best_tracker].copy() if best_tracker else pd.DataFrame()

        if args.debug:
            print(f"\n[DEBUG] fps={fps}")
            print(f"  bt_rows={len(bt)} trackers={sorted(bt['tracker'].astype(str).unique())}")
            print(f"  my_rows={len(my)} chosen_my={chosen}")

        bt_scores = mean_metrics(bt, args.score_metrics)
        my_scores = mean_metrics(my, args.score_metrics)
        bt_counts = mean_metrics(bt, args.count_metrics)
        my_counts = mean_metrics(my, args.count_metrics)

        plot_bar_scores(fps, my_scores, bt_scores, args.score_metrics, out_dir, args.title)
        plot_bar_counts(fps, my_counts, bt_counts, args.count_metrics, out_dir, args.title)

        bt_a, bt_c = mean_curve(bt, "HOTA")
        my_a, my_c = mean_curve(my, "HOTA")
        if bt_a.size or my_a.size:
            plot_hota_curve(fps, my_a, my_c, bt_a, bt_c, out_dir, args.title)

    print(f"\nSaved plots to: {out_dir.resolve()}")
    for p in sorted(out_dir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()