"""
Created on Wed Aug 13 06:12:01 2025

@author: tomtang
"""

#!/usr/bin/env python3
import os
import glob
import shutil
from datetime import datetime

import numpy as np
import h5py
import matplotlib.pyplot as plt

DATA_DIR = "./data"
PLOT_DIR = "./plots"
PATTERN = os.path.join(DATA_DIR, "*.h5")

EXPECTED_NELS = [4, 9, 16, 25, 36]   # Nel = Nel_1d^2
NEL_1D       = np.sqrt(np.array(EXPECTED_NELS, dtype=float))  # x-axis
K_LINES      = [1, 2, 3, 4, 5]       # rows
color_list   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

os.makedirs(PLOT_DIR, exist_ok=True)

def read_record(path):
    """Assumes exact filenames and all attributes/datasets present."""
    with h5py.File(path, "r") as f:
        g = f["NumericalSolution"]
        method = g.attrs["method"].decode() if isinstance(g.attrs["method"], bytes) else g.attrs["method"]
        nel    = int(g.attrs["Nel"])
        k      = int(g.attrs["k"])
        l2     = float(g["l2_error"][()])
        return method.upper(), nel, k, l2

def l2_matrix_row_major(method, records):
    """
    Returns matrix shaped (len(K_LINES), len(EXPECTED_NELS)) with C-order (row-major).
    Rows correspond to k = 1..5, columns to Nel_1d = 2..6 (via EXPECTED_NELS).
    """
    M = np.full((len(K_LINES), len(EXPECTED_NELS)), np.nan, order="C", dtype=float)
    for m, nel, kval, l2 in records:
        if m != method:
            continue
        i = K_LINES.index(kval)          # row (k)
        j = EXPECTED_NELS.index(nel)     # col (Nel)
        M[i, j] = l2
    return M

def plot_method(method, M):
    """
    Plot 5 lines (rows of M) vs Nel_1d with fixed colors and '-o'.
    Includes the theoretical slope guides you specified.
    """
    plt.figure(figsize=(6, 4))
    for i, k in enumerate(K_LINES):
        y = M[i, :]  # row i (k fixed)
        x = NEL_1D
        # filter out junk points, if any
        mask = np.isfinite(y) & (y > 0)
        plt.loglog(x[mask], y[mask], "-o", color=color_list[i],
                   label=f"k={k}", linewidth=1.6, markersize=5)

        # Slope guides anchored at first available point
        if np.any(mask):
            y0 = y[mask][0]
            if method in ("SIPG", "IIPG"):
                const_plotscale = (2.5 ** (i + 2)) * y0
                plt.loglog(x, const_plotscale * (1 / x) ** (i + 2),
                           "--o", color=color_list[i], markersize=3,
                           label=fr"$O(h^{i+2})$")
            else:  # NIPG
                const_plotscale = (2.0 ** (i + 2)) * y0
                plt.loglog(x, const_plotscale * (1 / x) ** (i + 1),
                           "--o", color=color_list[i], markersize=3,
                           label=fr"$O(h^{i+1})$")

    plt.xlabel("Nel_1d")
    plt.ylabel("vector l2 error")
    plt.title(f"{method} convergence")
    # plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(frameon=True, loc="right", bbox_to_anchor=(1.25, 0.5))
    plt.tight_layout()
    out_png = os.path.join(PLOT_DIR, f"convergence_L2_{method}.png")
    plt.savefig(out_png)
    print(f"Saved {out_png}")

def archive_data_dir():
    """
    Move all current contents of ./data into a timestamped subfolder
    like ./data/runat_YYMMDD_HHMM
    """
    ts_label = datetime.now().strftime("runat_%y%m%d_%H%M")
    archive_dir = os.path.join(DATA_DIR, ts_label)
    os.makedirs(archive_dir, exist_ok=True)

    # Move all files and subfolders except the archive_dir itself
    for name in os.listdir(DATA_DIR):
        src = os.path.join(DATA_DIR, name)
        if os.path.abspath(src) == os.path.abspath(archive_dir):
            continue
        shutil.move(src, archive_dir)

    print(f"Archived data files to {archive_dir}")

def main():
    files = sorted(glob.glob(PATTERN))
    records = [read_record(p) for p in files]

    # Build row-major matrices
    SIPG_l2_matrix  = l2_matrix_row_major("SIPG", records)
    NIPG_l2_matrix  = l2_matrix_row_major("NIPG", records)
    IIPG_L2_matrix  = l2_matrix_row_major("IIPG", records)

    # Save with explicit C-order
    np.savez(os.path.join(DATA_DIR, "l2_convergence_matrices.npz"),
             SIPG_l2_matrix=np.array(SIPG_l2_matrix, order="C"),
             NIPG_l2_matrix=np.array(NIPG_l2_matrix, order="C"),
             IIPG_L2_matrix=np.array(IIPG_L2_matrix, order="C"),
             Nels=np.array(EXPECTED_NELS),
             Nel_1d=NEL_1D,
             ks=np.array(K_LINES))

    # Plots (saved into ./plots)
    plot_method("SIPG", SIPG_l2_matrix)
    plot_method("IIPG", IIPG_L2_matrix)
    plot_method("NIPG", NIPG_l2_matrix)

    # Show all figures together
    plt.show()

    # Archive ./data into timestamped subfolder
    archive_data_dir()

    return [SIPG_l2_matrix, NIPG_l2_matrix, IIPG_L2_matrix]

if __name__ == "__main__":
    SIPG_l2_matrix, NIPG_l2_matrix, IIPG_L2_matrix = main()

