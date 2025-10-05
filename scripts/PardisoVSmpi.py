
import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def y_plain_formatter(val, _pos):
    if val >= 100:
        return f"{int(val):d}"
    elif val >= 10:
        return f"{val:.1f}".rstrip('0').rstrip('.')
    elif val >= 1:
        return f"{val:.2f}".rstrip('0').rstrip('.')
    else:
        return f"{val:.3f}".rstrip('0').rstrip('.')

def parse_k_from_name(path):
    m = re.search(r"_k=(\d+)", os.path.basename(path))
    return m.group(1) if m else None

def parse_method_from_name(path):
    # Capture method token that excludes underscores to avoid swallowing '_Nel'
    m = re.search(r"method=([A-Za-z0-9+-]+)", os.path.basename(path))
    return m.group(1) if m else "SIPG"

def parse_nel_from_name(path):
    m = re.search(r"Nel=(\d+)", os.path.basename(path))
    return m.group(1) if m else None

def power_two_ticks(xmin, xmax):
    xmin = max(1, int(np.floor(xmin)))
    xmax = int(np.ceil(xmax))
    ticks = []
    p = 0
    while 2**p < xmin:
        p += 1
    while 2**p <= xmax:
        ticks.append(2**p)
        p += 1
    return ticks

def main():
    ap = argparse.ArgumentParser(description="Plot Program Wall Time vs Threads for Pardiso (MKL) and MPI+JCG on a log-log chart.")
    ap.add_argument("--solver_csv", default="./testSolver_sp_walltime_vs_threads_testcase0_method=SIPG_Nel=65536_k=2.csv",
                    help="CSV for MKLPardiso (expects columns: threads, mean_wall_time)")
    ap.add_argument("--jcg_csv", default="./testJCG_walltime_vs_threads_testcase0_method=SIPG_Nel=65536_k=2.csv",
                    help="CSV for MPI+JCG (expects columns: 'NumCores=p (p0=2)', 'Program Wall time')")
    ap.add_argument("--out", default="../plots/walltime_vs_threads.png", help="Output PNG path")
    args = ap.parse_args()

    # Load data
    df_solver = pd.read_csv(args.solver_csv)
    df_jcg = pd.read_csv(args.jcg_csv)

    # Column mapping
    if not {"threads", "mean_wall_time"}.issubset(df_solver.columns):
        raise KeyError("MKLPardiso CSV must have columns: 'threads' and 'mean_wall_time'")
    if not {"NumCores=p (p0=2)", "Program Wall time"}.issubset(df_jcg.columns):
        raise KeyError("MPI+JCG CSV must have columns: 'NumCores=p (p0=2)' and 'Program Wall time'")

    df_solver = df_solver[["threads", "mean_wall_time"]].dropna()
    df_jcg = df_jcg[["NumCores=p (p0=2)", "Program Wall time"]].dropna()

    # Coerce numeric
    df_solver["threads"] = pd.to_numeric(df_solver["threads"], errors="coerce")
    df_solver["mean_wall_time"] = pd.to_numeric(df_solver["mean_wall_time"], errors="coerce")
    df_jcg["NumCores=p (p0=2)"] = pd.to_numeric(df_jcg["NumCores=p (p0=2)"], errors="coerce")
    df_jcg["Program Wall time"] = pd.to_numeric(df_jcg["Program Wall time"], errors="coerce")

    df_solver = df_solver.dropna().sort_values(by="threads")
    df_jcg = df_jcg.dropna().sort_values(by="NumCores=p (p0=2)")

    x1, y1 = df_solver["threads"].values, df_solver["mean_wall_time"].values
    x2, y2 = df_jcg["NumCores=p (p0=2)"].values, df_jcg["Program Wall time"].values

    # Prepare ticks
    xticks = power_two_ticks(min(x1.min(), x2.min()), max(x1.max(), x2.max()))

    # Y ticks (1-2-5 per decade), plain-number formatter
    ymin = float(min(y1.min(), y2.min()))
    ymax = float(max(y1.max(), y2.max()))
    lo_dec = int(np.floor(np.log10(ymin))) if ymin > 0 else -1
    hi_dec = int(np.ceil(np.log10(ymax)))
    y_ticks = []
    for d in range(lo_dec, hi_dec + 1):
        for base in (1, 2, 5):
            val = base * (10 ** d)
            if ymin*0.95 <= val <= ymax*1.05:
                y_ticks.append(val)
    y_ticks = sorted(set(y_ticks))

    # Parse method, k, and Nel from filenames
    method = parse_method_from_name(args.solver_csv) or parse_method_from_name(args.jcg_csv) or "SIPG"
    k = parse_k_from_name(args.solver_csv) or parse_k_from_name(args.jcg_csv) or "?"
    nel = parse_nel_from_name(args.solver_csv) or parse_nel_from_name(args.jcg_csv) or "?"

    # Plot
    plt.figure(figsize=(7,5))
    plt.loglog(x1, y1, marker="o", label="Gadi_intel8268_MKLPardiso")
    plt.loglog(x2, y2, marker="s", label="Gadi_intel8268_MPI+JCG")
    plt.xlabel("physical cores")
    plt.ylabel("Program Wall Time (s)")
    plt.title(f"DGSEM 2D {method}, order k={k}, Nel={nel}: Program Wall Time vs Physical Cores (log-log)")
    ax = plt.gca()
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(y_plain_formatter))
    plt.grid(True, which="both", linewidth=0.5, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    # Output naming pattern
    out_path = args.out
    # If the user left the default or empty, use the requested pattern
    if out_path is None or out_path.strip() == "" or out_path.endswith("walltime_vs_threads.png"):
        fname = f"testcase0_{method}_k{k}_Nel{nel}_walltime_vs_cores.png"
        out_dir = os.path.dirname(args.out) if args.out else "./plots"
        if out_dir == "":
            out_dir = "../plots"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, dpi=160)
    print(out_path)

if __name__ == "__main__":
    main()
