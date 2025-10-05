#!/usr/bin/env python3
import argparse, subprocess, sys, re, csv, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Sweep sigma0 and plot broken L2 error vs sigma0 (2D DGSEM).")
    p.add_argument("--exe", required=True, help="Path to DG 2D executable (e.g., ./testSolver_sp)")
    p.add_argument("--method", default="SIPG", choices=["SIPG","IIPG","NIPG","NIPG0"], help="DG method")
    p.add_argument("-k", type=int, required=True, help="Polynomial order k")
    p.add_argument("--nelx", type=int, default=4, help="Number of elements in x")
    p.add_argument("--nely", type=int, default=4, help="Number of elements in y")
    p.add_argument("--xla", type=float, default=0.0); p.add_argument("--xlb", type=float, default=1.0)
    p.add_argument("--yla", type=float, default=0.0); p.add_argument("--ylb", type=float, default=1.0)
    p.add_argument("--bc",  default="L=D,R=D,B=D,T=D", help='Boundary spec like "L=D,R=D,B=D,T=D" (Dirichlet/Neumann)')
    p.add_argument("--case", required=True, help="Path to JSON case file")
    p.add_argument("--write", choices=["w","nw"], default="nw", help="Write HDF5 outputs or not")
    p.add_argument("--samples", type=int, default=50, help="Number of sigma0 samples (default 50)")
    p.add_argument("--sigma_max_factor", type=float, default=3.0, help="Sweep up to factor * (k*k) (default 3.0)")
    p.add_argument("--outdir", default="plots", help="Output directory for CSV and PNG")
    p.add_argument("--logy", action="store_true", help="Use log scale on Y")
    return p.parse_args()

BROKEN_L2_PAT = re.compile(r"relative\s+broken\s+L2\s*=\s*([0-9eE\+\-\.]+)")

def run_one(args, sigma0):
    cmd = [
        args.exe,
        args.write,
        args.method,
        str(args.k),
        f"{sigma0:.16g}",
        str(args.nelx),
        str(args.nely),
        f"{args.xla:.16g}", f"{args.xlb:.16g}",
        f"{args.yla:.16g}", f"{args.ylb:.16g}",
        args.bc,
        args.case,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Command failed (sigma0={sigma0}):\n{e.stdout}", file=sys.stderr)
        return math.nan, e.stdout

    out = res.stdout
    m = BROKEN_L2_PAT.search(out)
    if not m:
        print(f"[WARN] Could not find 'relative broken L2' in output (sigma0={sigma0}).", file=sys.stderr)
        return math.nan, out
    return float(m.group(1)), out

def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    case_name = Path(args.case).stem
    sigma_star = args.k * args.k
    sigmas = np.linspace(0.0, args.sigma_max_factor * sigma_star, args.samples)
    rows = []
    all_stdout = []

    print(f"Sweeping sigma0 in [0, {args.sigma_max_factor}*k^2] with {args.samples} samples...")
    for s in sigmas:
        val, out = run_one(args, s)
        rows.append((s, val))
        all_stdout.append({"sigma0": s, "stdout": out})

    # Save CSV
    csv_path = Path(args.outdir) / f"scanSigma_{case_name}_k{args.k}_{args.method}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma0", "broken_L2_rel"])
        for s, v in rows:
            w.writerow([f"{s:.16g}", f"{v:.16g}" if not math.isnan(v) else "nan"])
    print(f"Wrote {csv_path}")

    # Optional: save the raw outputs for debugging
    txt_path = Path(args.outdir) / f"scanSigma_{case_name}_k{args.k}_{args.method}_stdout.txt"
    with open(txt_path, "w") as f:
        for entry in all_stdout:
            f.write(f"===== sigma0 = {entry['sigma0']:.16g} =====\n")
            f.write(entry["stdout"])
            if not entry["stdout"].endswith("\n"):
                f.write("\n")
    print(f"Wrote {txt_path}")

    # Plot
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]

    plt.figure(figsize=(7,4.2))
    plt.plot(xs, ys, marker="o", linewidth=1.0, markersize=3)  # no explicit color per instruction
    # vertical red dashed at k*k
    plt.axvline(x=sigma_star, color="red", linestyle="--", linewidth=1.2, label=rf"$k^2$={sigma_star}")
    plt.xlabel(r"$\sigma_0$")
    plt.ylabel("relative broken L2")
    plt.title(f"DGSEM 2D {args.method}, k={args.k}, Nelx={args.nelx}, Nely={args.nely}")
    if args.logy:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend()
    png_path = Path(args.outdir) / f"scanSigma_{case_name}_k{args.k}_{args.method}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    print(f"Wrote {png_path}")

if __name__ == "__main__":
    main()
