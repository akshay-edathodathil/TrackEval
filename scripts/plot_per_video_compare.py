#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SCORE_METRICS_DEFAULT = ["HOTA(0)", "LocA(0)", "HOTA___AUC", "DetA___AUC", "AssA___AUC", "MOTA", "IDF1", "CLR_F1"]
COUNT_METRICS_DEFAULT = ["IDSW", "Frag"]


def normalize_scores(vals: np.ndarray) -> np.ndarray:
    """Convert 0..100 style to 0..1 if needed."""
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return vals
    if np.nanmedian(finite) > 1.5 or np.nanmax(finite) > 2.0:
        return vals / 100.0
    return vals


def read_detailed_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if df.empty:
        return df
    if "sequence" not in df.columns:
        df = df.rename(columns={df.columns[0]: "sequence"})
    df["sequence"] = df["sequence"].astype(str)
    return df


def get_row(df: pd.DataFrame, seq: str) -> pd.Series:
    m = df["sequence"].astype(str) == seq
    if not m.any():
        raise KeyError(f"Sequence '{seq}' not found in detailed CSV.")
    return df.loc[m].iloc[0]


def list_pig_seqs(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    pigs = [f"pig_{i}" for i in range(1, 6)]
    sa = set(df_a["sequence"].astype(str).tolist())
    sb = set(df_b["sequence"].astype(str).tolist())
    if all(p in sa and p in sb for p in pigs):
        return pigs
    # fallback: intersection of non-COMBINED rows
    inter = sorted((sa & sb) - {"COMBINED"})
    return [s for s in inter if not s.upper().startswith("COMBINED")]


def extract_curve(row: pd.Series, prefix: str = "HOTA") -> Tuple[np.ndarray, np.ndarray]:
    pairs = []
    for c in row.index:
        if not isinstance(c, str):
            continue
        if not c.startswith(prefix + "___"):
            continue
        suf = c.split("___", 1)[1]
        if suf.isdigit():
            v = pd.to_numeric(row[c], errors="coerce")
            pairs.append((int(suf), float(v) if pd.notna(v) else np.nan))
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        return np.array([]), np.array([])
    x = np.array([k for k, _ in pairs], dtype=int)
    y = normalize_scores(np.array([v for _, v in pairs], dtype=float))
    return x, y


def plot_bar_scores(seq: str, bt: pd.Series, my: pd.Series, metrics: List[str], out_dir: Path, title: str):
    metrics = [m for m in metrics if (m in bt.index) or (m in my.index)]
    if not metrics:
        return

    bt_vals = np.array([pd.to_numeric(bt.get(m, np.nan), errors="coerce") for m in metrics], dtype=float)
    my_vals = np.array([pd.to_numeric(my.get(m, np.nan), errors="coerce") for m in metrics], dtype=float)
    bt_vals = normalize_scores(bt_vals)
    my_vals = normalize_scores(my_vals)

    x = np.arange(len(metrics))
    w = 0.38
    plt.figure(figsize=(max(10, len(metrics) * 1.25), 5))
    plt.bar(x - w/2, bt_vals, width=w, label="ByteTrack")
    plt.bar(x + w/2, my_vals, width=w, label="FAWW / Best sweep")
    plt.xticks(x, metrics, rotation=35, ha="right")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.title(f"{title} — {seq} (scores)")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_scores.png", dpi=200)
    plt.close()


def plot_bar_counts(seq: str, bt: pd.Series, my: pd.Series, metrics: List[str], out_dir: Path, title: str):
    metrics = [m for m in metrics if (m in bt.index) or (m in my.index)]
    if not metrics:
        return

    bt_vals = np.array([pd.to_numeric(bt.get(m, np.nan), errors="coerce") for m in metrics], dtype=float)
    my_vals = np.array([pd.to_numeric(my.get(m, np.nan), errors="coerce") for m in metrics], dtype=float)

    x = np.arange(len(metrics))
    w = 0.38
    plt.figure(figsize=(max(7, len(metrics) * 1.7), 5))
    plt.bar(x - w/2, bt_vals, width=w, label="ByteTrack")
    plt.bar(x + w/2, my_vals, width=w, label="FAWW / Best sweep")
    plt.xticks(x, metrics, rotation=35, ha="right")
    plt.ylabel("Count")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.title(f"{title} — {seq} (counts)")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_counts.png", dpi=200)
    plt.close()


def plot_hota_curve(seq: str, bt: pd.Series, my: pd.Series, out_dir: Path, title: str):
    bx, by = extract_curve(bt, "HOTA")
    mx, myc = extract_curve(my, "HOTA")
    if bx.size == 0 and mx.size == 0:
        return

    plt.figure(figsize=(10, 5))
    if bx.size:
        plt.plot(bx, by, marker="o", label="ByteTrack")
    if mx.size:
        plt.plot(mx, myc, marker="o", label="FAWW / Best sweep")
    plt.xlabel("Alpha (e.g., 5 = 0.05)")
    plt.ylabel("HOTA")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"{title} — {seq} (HOTA curve)")
    plt.tight_layout()
    plt.savefig(out_dir / "hota_curve.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers_root", type=str, required=True,
                    help="e.g. data/trackers/mot_challenge/Pig30fps-train")
    ap.add_argument("--bytetrack", type=str, required=True,
                    help="tracker folder name (must contain pedestrian_detailed.csv)")
    ap.add_argument("--mytracker", type=str, required=True,
                    help="tracker folder name (must contain pedestrian_detailed.csv)")
    ap.add_argument("--out_root", type=str, default="/Users/akshay/Downloads/pig/results/_per_video_30fps",
                    help="where to save per-video folders")
    ap.add_argument("--class_name", type=str, default="pedestrian")
    ap.add_argument("--title", type=str, default="Pig 30fps: FAWW vs ByteTrack")
    ap.add_argument("--score_metrics", type=str, nargs="+", default=SCORE_METRICS_DEFAULT)
    ap.add_argument("--count_metrics", type=str, nargs="+", default=COUNT_METRICS_DEFAULT)
    args = ap.parse_args()

    tr = Path(args.trackers_root)

    bt_csv = tr / args.bytetrack / f"{args.class_name}_detailed.csv"
    my_csv = tr / args.mytracker / f"{args.class_name}_detailed.csv"

    if not bt_csv.exists():
        raise SystemExit(f"Missing: {bt_csv}")
    if not my_csv.exists():
        raise SystemExit(f"Missing: {my_csv}")

    bt_df = read_detailed_csv(bt_csv)
    my_df = read_detailed_csv(my_csv)

    seqs = list_pig_seqs(bt_df, my_df)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for seq in seqs:
        bt_row = get_row(bt_df, seq)
        my_row = get_row(my_df, seq)

        out_dir = out_root / seq
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_bar_scores(seq, bt_row, my_row, args.score_metrics, out_dir, args.title)
        plot_bar_counts(seq, bt_row, my_row, args.count_metrics, out_dir, args.title)
        plot_hota_curve(seq, bt_row, my_row, out_dir, args.title)

        # Save compact table with deltas
        cols = ["sequence"] + list(dict.fromkeys(args.score_metrics + args.count_metrics))
        cols = [c for c in cols if c in bt_row.index or c in my_row.index]
        rec_bt = {c: bt_row.get(c, np.nan) for c in cols}
        rec_my = {c: my_row.get(c, np.nan) for c in cols}
        rec_dt = {c: (pd.to_numeric(rec_my.get(c, np.nan), errors="coerce") -
                      pd.to_numeric(rec_bt.get(c, np.nan), errors="coerce"))
                  for c in cols if c != "sequence"}
        rec_dt["sequence"] = seq

        pd.DataFrame([
            {"method": "ByteTrack", **rec_bt},
            {"method": "FAWW/BestSweep", **rec_my},
            {"method": "Delta(My-BT)", **rec_dt},
        ]).to_csv(out_dir / "metrics_rows.csv", index=False)

    print(f"Saved per-video plots under: {out_root.resolve()}")


if __name__ == "__main__":
    main()