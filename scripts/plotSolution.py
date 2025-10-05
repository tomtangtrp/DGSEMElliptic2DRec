#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# ---------- small helpers ----------
def read_attr(attrs, key, default=None):
    if key in attrs:
        v = attrs[key]
        try:
            if isinstance(v, (bytes, bytearray)):
                try:
                    return v.decode()
                except Exception:
                    return str(v)
            if hasattr(v, "shape") and v.shape == ():
                return v[()]
            if hasattr(v, "item"):
                return v.item()
            return v
        except Exception:
            return v
    return default

def best_factor_pair(N, aspect):
    """Pick (Nelx, Nely) s.t. Nelx*Nely=N and Nelx/Nely ~ aspect (x wider if aspect>1)."""
    best = (N, 1)
    best_err = float("inf")
    for nely in range(1, N + 1):
        if N % nely:
            continue
        nelx = N // nely
        err = abs((nelx / nely) - aspect)
        if err < best_err:
            best = (nelx, nely)
            best_err = err
    return best

def element_xy(ex, ey, Nelx, Nely, xLa, xLb, yLa, yLb, k):
    hx = (xLb - xLa) / Nelx
    hy = (yLb - yLa) / Nely
    x0 = xLa + ex * hx
    y0 = yLa + ey * hy
    k1 = k + 1
    # simple uniform nodes per element for visualization
    xloc = np.linspace(x0, x0 + hx, k1)
    yloc = np.linspace(y0, y0 + hy, k1)
    X, Y = np.meshgrid(xloc, yloc, indexing="xy")
    return X, Y

def reshape_tiles(vec, Nelx, Nely, k):
    k1 = k + 1
    Ne = Nelx * Nely
    arr = np.asarray(vec).reshape(Ne, k1, k1)  # row-major, x-fastest per your writer
    return arr

# ---------- main plotting ----------
def main():
    ap = argparse.ArgumentParser(description="Plot DG2D rectangular solution (numerical + exact overlay, and numerical-only).")
    ap.add_argument("h5file", help="Path to HDF5 written by the solver")
    ap.add_argument("--nelx", type=int, help="Override Nelx (optional)")
    ap.add_argument("--nely", type=int, help="Override Nely (optional)")
    ap.add_argument("--alpha", type=float, default=0.75, help="Surface transparency [0..1]")
    ap.add_argument("--elev", type=float, default=30.0, help="View elev")
    ap.add_argument("--azim", type=float, default=-60.0, help="View azim")
    ap.add_argument("--save", help="Save the overlay figure; numerical-only saves as <name>_numonly.png")
    args = ap.parse_args()

    with h5py.File(args.h5file, "r") as hf:
        # --- Grid / domain
        G = hf["/Grid"]
        xLa = float(read_attr(G.attrs, "xLa"))
        xLb = float(read_attr(G.attrs, "xLb"))
        yLa = float(read_attr(G.attrs, "yLa"))
        yLb = float(read_attr(G.attrs, "yLb"))
        Lx, Ly = (xLb - xLa), (yLb - yLa)
        aspect = (Lx / Ly) if Ly != 0 else 1.0

        # --- Numerical solution + meta
        NG = hf["/NumericalSolution"]
        Nel = int(read_attr(NG.attrs, "Nel"))
        k   = int(read_attr(NG.attrs, "k"))
        method = read_attr(NG.attrs, "method", "")
        sigma0 = read_attr(NG.attrs, "sigma0", None)
        u_h = np.array(NG["u_h"])
        k1 = k + 1
        assert u_h.size == Nel * k1 * k1, f"u_h length {u_h.size} != Nel*(k+1)^2={Nel*k1*k1}"

        # Determine Nelx/Nely:
        if args.nelx and args.nely:
            Nelx, Nely = args.nelx, args.nely
        else:
            Nelx_attr = read_attr(NG.attrs, "Nelx")
            Nely_attr = read_attr(NG.attrs, "Nely")
            if Nelx_attr is not None and Nely_attr is not None:
                Nelx, Nely = int(Nelx_attr), int(Nely_attr)
            else:
                Nel_x_grid = read_attr(G.attrs, "Nel_x")
                Nel_y_grid = read_attr(G.attrs, "Nel_y")
                if Nel_x_grid is not None and Nel_y_grid is not None:
                    Nelx, Nely = int(Nel_x_grid), int(Nel_y_grid)
                else:
                    Nelx, Nely = best_factor_pair(Nel, aspect)

        Utiles = reshape_tiles(u_h, Nelx, Nely, k)

        # --- Exact (optional)
        EXtiles = None
        if "/ExactSolution" in hf and "u_exact" in hf["/ExactSolution"]:
            u_ex = np.array(hf["/ExactSolution/u_exact"])
            if u_ex.size == u_h.size:
                EXtiles = reshape_tiles(u_ex, Nelx, Nely, k)
            else:
                print(f"[warn] exact length {u_ex.size} != numerical {u_h.size}; plotting numerical only.")

        # =========================
        # Figure 1: overlay (num + exact)
        # =========================
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=args.elev, azim=args.azim)

        # numerical first (blue)
        for ey in range(Nely):
            for ex in range(Nelx):
                e = ey * Nelx + ex
                X, Y = element_xy(ex, ey, Nelx, Nely, xLa, xLb, yLa, yLb, k)
                Z = Utiles[e]
                ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False,
                                rstride=1, cstride=1, color="b", alpha=args.alpha)

        # exact overlay (yellow), if available
        if EXtiles is not None:
            for ey in range(Nely):
                for ex in range(Nelx):
                    e = ey * Nelx + ex
                    X, Y = element_xy(ex, ey, Nelx, Nely, xLa, xLb, yLa, yLb, k)
                    Z = EXtiles[e]
                    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False,
                                    rstride=1, cstride=1, color="y", alpha=args.alpha)
            ax.legend([Patch(facecolor="b", alpha=args.alpha, label="Numerical"),
                       Patch(facecolor="y", alpha=args.alpha, label="Exact")],
                      ["Numerical", "Exact"], loc="upper right")

        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u")
        title_bits = [f"{method}", f"k={k}", f"Nel={Nelx}×{Nely}", f"[{xLa},{xLb}]×[{yLa},{yLb}]"]
        if sigma0 is not None:
            try:
                title_bits.insert(1, f"σ₀={float(sigma0)}")
            except Exception:
                title_bits.insert(1, f"sigma0={sigma0}")
        ax.set_title("  ".join(title_bits))

        # =========================
        # Figure 2: numerical-only (blue)
        # =========================
        fig2 = plt.figure(figsize=(9, 7))
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.view_init(elev=args.elev, azim=args.azim)

        for ey in range(Nely):
            for ex in range(Nelx):
                e = ey * Nelx + ex
                X, Y = element_xy(ex, ey, Nelx, Nely, xLa, xLb, yLa, yLb, k)
                Z = Utiles[e]
                ax2.plot_surface(X, Y, Z, linewidth=0, antialiased=False,
                                 rstride=1, cstride=1, color="b", alpha=args.alpha)

        ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("u")
        ax2.set_title("Numerical only  " + "  ".join(title_bits))

        # ---------- save/show ----------
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
            root, ext = os.path.splitext(args.save)
            fig2.savefig(root + "_numonly" + (ext if ext else ".png"),
                         dpi=150, bbox_inches="tight")
        plt.show()

if __name__ == "__main__":
    main()
