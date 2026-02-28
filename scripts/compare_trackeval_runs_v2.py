from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def find_tracker_dirs(trackers_root: Path):
    detailed = {p.parent for p in trackers_root.rglob("pedestrian_detailed.csv")}
    summary = {p.parent for p in trackers_root.rglob("pedestrian_summary.txt")}
    return sorted(detailed | summary)

def read_combined_from_detailed(csv_path: Path):
    df = pd.read_csv(csv_path)

    seq_col = None
    for c in ["seq", "sequence", "Sequence", "name"]:
        if c in df.columns:
            seq_col = c
            break
    if seq_col is None:
        seq_col = df.columns[0]

    comb = df[df[seq_col].astype(str).str.upper().str.contains("COMBINED", na=False)]
    if comb.empty:
        comb = df[df[seq_col].astype(str).str.upper().isin(["ALL", "OVERALL"])]

    if comb.empty:
        raise RuntimeError(f"No COMBINED row in {csv_path}")

    row = comb.iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        if k == seq_col:
            continue
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out

def read_combined_any(tracker_dir: Path):
    detailed = tracker_dir / "pedestrian_detailed.csv"
    if detailed.exists():
        try:
            return read_combined_from_detailed(detailed)
        except Exception:
            pass
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trackers_root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--sort_by", default="HOTA(0)")
    args = parser.parse_args()

    trackers_root = Path(args.trackers_root).resolve()
    if not trackers_root.exists():
        raise SystemExit(f"trackers_root not found: {trackers_root}")

    tracker_dirs = find_tracker_dirs(trackers_root)
    if not tracker_dirs:
        raise SystemExit(
            f"No trackers parsed under: {trackers_root}\n"
            f"Expected pedestrian_detailed.csv or pedestrian_summary.txt somewhere below it."
        )

    rows = []
    for td in tracker_dirs:
        metrics = read_combined_any(td)
        if metrics is None:
            continue
        metrics["tracker"] = td.name
        rows.append(metrics)

    if not rows:
        raise SystemExit(
            "Found tracker dirs, but failed to read any combined metrics. "
            "Check that pedestrian_detailed.csv contains a COMBINED row."
        )

    df = pd.DataFrame(rows).set_index("tracker")

    if args.baseline not in df.index:
        raise SystemExit(
            f"Baseline '{args.baseline}' not found. Available:\n" +
            "\n".join(df.index)
        )

    base = df.loc[args.baseline]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[f"Δ{col}_vs_{args.baseline}"] = df[col] - base[col]

    sort_col = args.sort_by
    if sort_col not in df.columns:
        for alt in ["HOTA", "IDF1", "MOTA"]:
            if alt in df.columns:
                sort_col = alt
                break

    df = df.sort_values(sort_col, ascending=False)

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)

    print("\nTop 10 trackers:")
    print(df.head(10).to_string())

    print(f"\nWrote: {out_csv}")

if __name__ == "__main__":
    main()