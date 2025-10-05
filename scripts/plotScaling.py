#!/usr/bin/env python3
# Plot log-log OMP scaling from one or many CSVs produced by ompScaling.py.
# - One file  -> single-curve plot
# - Many files -> combined plot (requires identical suffix after first '_')
# - X ticks show exact tested thread counts

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def read_series(csv_path: Path):
    threads, means = [], []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = int(row["threads"])
                m = float(row["mean_wall_time"])
            except (KeyError, ValueError):
                continue
            if str(m).lower() == "nan":
                continue
            threads.append(t)
            means.append(m)
    z = sorted(zip(threads, means))
    return [t for t, _ in z], [m for _, m in z]

def split_prefix_suffix(p: Path):
    stem = p.stem
    if "_" not in stem:
        return stem, ""
    i = stem.find("_")
    return stem[:i], stem[i+1:]

def main():
    ap = argparse.ArgumentParser(
        description="Plot log-log OMP scaling from CSV(s) produced by ompScaling.py"
    )
    ap.add_argument("csvs", nargs="+",
                    help="CSV files like '14900k_walltime_vs_threads_...csv'")
    ap.add_argument("--out", help="Output PNG filename (optional)")
    ap.add_argument("--title-prefix", default="OMP scaling",
                    help="Title prefix (optional)")
    args = ap.parse_args()

    files = [Path(x) for x in args.csvs]
    if len(files) < 1:
        raise SystemExit("Provide at least one CSV file.")

    # Build labels (hardware prefix) and suffixes (everything after first '_')
    hw_names, suffixes = [], []
    for p in files:
        hw, suf = split_prefix_suffix(p)
        hw_names.append(hw)
        suffixes.append(suf)

    # If multiple files, require identical suffix after the first underscore
    if len(files) > 1:
        suffix_ref = suffixes[0]
        if any(s != suffix_ref for s in suffixes):
            msg = "ERROR: All inputs must have identical suffixes after the first underscore.\n"
            msg += "\n".join(f"- {p.name} -> suffix='{s}'" for p, s in zip(files, suffixes))
            raise SystemExit(msg)
    else:
        suffix_ref = suffixes[0]  # single-file case

    plt.figure(figsize=(7, 5))
    all_threads = set()

    for p, hw in zip(files, hw_names):
        th, mu = read_series(p)
        if not th:
            print(f"[WARN] No valid data in {p}")
            continue
        all_threads.update(th)
        label = hw if len(files) > 1 else p.stem  # single-file: more specific label
        plt.loglog(th, mu, "--o", label=label)

    # Exact core ticks: union of all tested counts
    xt = sorted(all_threads)
    ax = plt.gca()
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(xt)
    ax.set_xticklabels([str(t) for t in xt])

    plt.xlabel("Threads")
    plt.ylabel("Wall time (seconds)")
    title_suffix = suffix_ref if len(files) > 1 else files[0].stem
    plt.title(f"{args.title_prefix}: {title_suffix}")
    plt.grid(True, which="both", alpha=0.3)
    if len(files) > 1:
        plt.legend()

    outpng = args.out or (f"scaling_{title_suffix}.png")
    plt.savefig(outpng, dpi=150, bbox_inches="tight")
    print(f"Wrote {outpng}")

if __name__ == "__main__":
    main()

