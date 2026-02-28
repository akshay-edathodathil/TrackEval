from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def find_tracker_dirs(trackers_root: Path) -> list[Path]:
    # Tracker output dirs look like: <tracker_dir>/pedestrian_detailed.csv or pedestrian_summary.txt
    # We search recursively and then map back to the parent (tracker dir).
    detailed = {p.parent for p in trackers_root.rglob("pedestrian_detailed.csv")}
    summary = {p.parent for p in trackers_root.rglob("pedestrian_summary.txt")}
    tracker_dirs = sorted(detailed | summary)
    return tracker_dirs

def read_combined_from_detailed(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    # identify sequence column (varies by TrackEval versions)
    seq_col = None
    for c in ["seq", "sequence", "Sequence", "name"]:
        if c in df.columns:
            seq_col = c
            break
    if seq_col is None:
        # fallback: assume first column is sequence-like
        seq_col = df.columns[0]

    # find combined row
    comb = df[df[seq_col].astype(str).str.upper().str.contains("COMBINED", na=False)]
    if comb.empty:
        # sometimes it's "ALL" or "OVERALL"
        comb = df[df[seq_col].astype(str).str.upper().isin(["ALL", "OVERALL"])]
    if comb.empty:
        raise RuntimeError(f"No COMBINED row in {csv_path}")

    row = comb.iloc[0].to_dict()
    # normalize: keep numeric columns only (except seq col)
    out = {}
    for k, v in row.items():
        if k == seq_col:
            continue
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out

def read_combined_any(tracker_dir: Path) -> dict | None:
    detailed = tracker_dir / "pedestrian_detailed.csv"
    if detailed.exists():
        try:
            return read_combined_from_detailed(detailed)
        except Exception:
            pass
    # If no detailed csv, we could parse summary.txt, but your runs DO create detailed csv.
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers_root", required=True)
    ap.add_argument("--baseline", required=True, help="baseline tracker folder name")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sort_by", default="HOTA(0)")
    args = ap.parse_args()

    trackers_root = Path(args.trackers_root).resolve()
    if not trackers_root.exists():
        raise SystemExit(f"trackers_root not found: {trackers_root}")

    tracker_dirs = find_tracker_dirs(trackers_root)
    if not tracker_dirs:
        raise SystemExit(f"No trackers parsed under: {trackers_root}\n"
                         f"Expected pedestrian_detailed.csv or pedestrian_summary.txt somewhere below it.")

    rows = []
    for td in tracker_dirs:
        tracker_name = td.name
        metrics = read_combined_any(td)
        if metrics is None:
            continue
        metrics["tracker"] = tracker_name
        rows.append(metrics)

    if not rows:
        raise SystemExit("Found tracker dirs, but failed to read any combined metrics. "
                         "Check that pedestrian_detailed.csv contains a COMBINED row.")

    df = pd.DataFrame(rows).set_index("tracker")

    if args.baseline not in df.index:
        raise SystemExit(f"Baseline '{args.baseline}' not found. Available:\n" + "\n".join(df.index))

    base = df.loc[args.baseline]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[f"Δ{col}_vs_{args.baseline}"] = df[col] - base[col]

    sort_col = args.sort_by
    if sort_col not in df.columns:
        # try common alternatives
        for alt in ["HOTA", "IDF1", "MOTA"]:
            if alt in df.columns:
                sort_col = alt
                break

    df = df.sort_values(sort_col, ascending=False)
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)

    # print top 10 with the most relevant columns if present
    cols_wanted = [c for c in ["HOTA(0)", "HOTA", "IDF1", "MOTA",
                              f"ΔHOTA(0)_vs_{args.baseline}",
                              f"ΔIDF1_vs_{args.baseline}",
                              f"ΔMOTA_vs_{args.baseline}"] if c in df.columns]
    print("\nTop 10 (sorted by %s):" % sort_col)
    print(df[cols_wanted].head(10).to_string())

    print(f"\nWrote: {out_csv}")

if __name__ == "__main__":
    main()