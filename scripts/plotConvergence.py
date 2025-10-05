#!/usr/bin/env python3
# Convergence plots for DG2DRec: three metrics with dashed slope guides.
# Usage: python3 plotConvergence.py <case_name_or_json> <METHOD>
# Example: python3 plotConvergence.py testcase1 SIPG
#
# Expects HDF5 files:
#   data/DG2DRec_<case>_method=<METHOD>_Nel=<Nel>_k=<k>_sigma0=<sigma0(k)>.h5
# with datasets under /NumericalSolution:
#   l2_error, broken_L2_rel, broken_H1_rel
#
# Nel is restricted to [4, 9, 16, 25, 36]; k to [1..5].

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR       = "./data" #pwd is same as bash so no need ../
PLOT_DIR       = "./plots"
#EXPECTED_NELS  = [4, 9, 16, 25, 36]     # Nel = Nel_1d^2
EXPECTED_NELS  = [4, 16, 64, 256, 1024]     # Nel = Nel_1d^2
K_LINES        = [1, 2, 3, 4, 5]        # polynomial orders
NEL_1D         = np.sqrt(np.array(EXPECTED_NELS, dtype=float))
ALLOWED        = {"SIPG", "IIPG", "NIPG", "NIPG0"}

# Old-style fixed colors and dashed guide style
color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def sigma0_for(method: str, k: int) -> int:
    m = method.upper()
    if m in ("SIPG", "IIPG"):
        return 2 * k * k + 1
    if m == "NIPG":
        return 1
    if m == "NIPG0":
        return 0
    return 0

def h5name(case_stem: str, method: str, Nel: int, k: int) -> str:
    s0 = sigma0_for(method, k)
    return os.path.join(
        DATA_DIR,
        f"DG2DRec_{case_stem}_method={method}_Nel={Nel}_k={k}_sigma0={s0}.h5"
    )

def _read_scalar_dset(group, name, default=np.nan):
    if name not in group:
        return default
    try:
        arr = np.array(group[name])
        return float(arr[()] if arr.shape == () else arr.squeeze()[()])
    except Exception:
        return default

def read_three_errors(h5path: str):
    """
    Read relative errors from /NumericalSolution datasets:
      l2_error -> relative vector ℓ2
      broken_L2_rel
      broken_H1_rel
    Returns (vec_rel, l2_rel, h1_rel) with NaN on failure.
    """
    try:
        with h5py.File(h5path, "r") as f:
            g = f["NumericalSolution"]
            v2 = _read_scalar_dset(g, "l2_error")
            l2 = _read_scalar_dset(g, "broken_L2_rel")
            h1 = _read_scalar_dset(g, "broken_H1_rel")
            return v2, l2, h1
    except Exception:
        return np.nan, np.nan, np.nan

def collect_matrix(case_stem: str, method: str, metric: str) -> np.ndarray:
    """
    Build a (len(K_LINES) x len(EXPECTED_NELS)) matrix of the chosen metric:
      metric ∈ {"vec", "l2", "h1"}.
    Rows -> k (1..5); Cols -> Nel (4,9,16,25,36).
    """
    M = np.full((len(K_LINES), len(EXPECTED_NELS)), np.nan, dtype=float)
    for i, k in enumerate(K_LINES):
        for j, Nel in enumerate(EXPECTED_NELS):
            p = h5name(case_stem, method, Nel, k)
            v2, l2, h1 = read_three_errors(p)
            M[i, j] = v2 if metric == "vec" else (l2 if metric == "l2" else h1)
    return M

def plot_family(case_stem: str, method: str, metric: str, M: np.ndarray, ylabel: str, out_png: str):
    """
    One figure with 5 curves (k = 1..5) vs Nel_1d on log-log axes.
    Dashed slope guides:
      Base order = (k+1) for SIPG/IIPG, k for NIPG/NIPG0.
      For H1 plots ONLY, we drop one order: ord_H1 = base_order - 1.
    """
    plt.figure(figsize=(6, 4))
    for i, k in enumerate(K_LINES):
        y = M[i, :]
        x = NEL_1D
        mask = np.isfinite(y) & (y > 0)

        # solid data line
        plt.loglog(x[mask], y[mask], "-o", color=color_list[i],
                   label=f"k={k}", linewidth=1.6, markersize=5)

        # dashed guide line (order-of-accuracy cue)
        if np.any(mask):
            y0 = y[mask][0]

            base_order = (k + 1) if method in ("SIPG") else k
            ord_ = base_order - 1 if method in ("SIPG") and metric == "h1" else base_order

            # keep old-style scaling constants
            scale = (2.5 if method in ("SIPG", "IIPG") else 2.0) ** (i + 2)
            guide = scale * y0 * (1.0 / x) ** ord_
            plt.loglog(x, guide, "--o", color=color_list[i], markersize=3,
                       label=fr"$O(h^{{{ord_}}})$")

    plt.xlabel("Nel_1d")
    plt.ylabel(ylabel)
    plt.title(f"{method}, case={case_stem}")
    # place legend outside to the right (old style)
    plt.legend(frameon=True, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_png}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 plotConvergence.py <case_name_or_json> <METHOD>")
        sys.exit(1)

    case_in  = sys.argv[1]
    method   = sys.argv[2].upper().replace(" ", "")  # allow "NIPG 0"
    if method not in ALLOWED:
        print("METHOD must be one of SIPG|IIPG|NIPG|NIPG0", file=sys.stderr)
        sys.exit(1)

    case_stem = os.path.splitext(os.path.basename(case_in))[0]

    # Build matrices from HDF5 datasets
    M_vec = collect_matrix(case_stem, method, "vec")  # /NumericalSolution/l2_error
    M_l2  = collect_matrix(case_stem, method, "l2")   # /NumericalSolution/broken_L2_rel
    M_h1  = collect_matrix(case_stem, method, "h1")   # /NumericalSolution/broken_H1_rel

    # Three plots (each: 5 curves across Nel list)
    plot_family(case_stem, method, "vec", M_vec, r"relative vector $\ell_2$",
                os.path.join(PLOT_DIR, f"conv_vectorL2rel_{case_stem}_{method}.png"))
    plot_family(case_stem, method, "l2",  M_l2,  r"relative broken $L^2$",
                os.path.join(PLOT_DIR, f"conv_brokenL2rel_{case_stem}_{method}.png"))
    plot_family(case_stem, method, "h1",  M_h1,  r"relative broken $H^1$ (semi)",
                os.path.join(PLOT_DIR, f"conv_brokenH1rel_{case_stem}_{method}.png"))

    plt.show()

if __name__ == "__main__":
    main()
