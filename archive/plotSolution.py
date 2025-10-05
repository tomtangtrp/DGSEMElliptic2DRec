#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def read_attr(attrs, key, default=None):
    if key in attrs:
        v = attrs[key]
        if isinstance(v, (bytes, bytearray)):
            try: return v.decode()
            except: return str(v)
        try:
            return v.item() if hasattr(v, "item") else v
        except Exception:
            return v
    return default

def best_factor_pair(N, aspect):
    """Pick (Nelx, Nely) s.t. Nelx*Nely=N and Nelx/Nely ~ aspect."""
    best = (N, 1)
    best_err = float("inf")
    for nely in range(1, N+1):
        if N % nely: continue
        nelx = N // nely
        err = abs((nelx / nely) - aspect)
        if err < best_err:
            best = (nelx, nely); best_err = err
    return best

def element_xy(ex, ey, Nelx, Nely, xLa, xLb, yLa, yLb, k):
    hx = (xLb - xLa) / Nelx
    hy = (yLb - yLa) / Nely
    x0 = xLa + ex * hx
    y0 = yLa + ey * hy
    k1 = k + 1
    # uniform nodes for visualization
    xloc = np.linspace(x0, x0 + hx, k1)
    yloc = np.linspace(y0, y0 + hy, k1)
    X, Y = np.meshgrid(xloc, yloc, indexing="xy")
    return X, Y

def reshape_tiles(vec, Nelx, Nely, k):
    k1 = k + 1
    Ne = Nelx * Nely
    arr = np.asarray(vec).reshape(Ne, k1, k1)  # C-order: x fastest
    return arr

def plot_both_surfaces(ax, Utiles, EXtiles, Nelx, Nely, xLa, xLb, yLa, yLb, k,
                       title, num_color="C0", ex_color="C1", alpha=0.75):
    # numerical first
    for ey in range(Nely):
        for ex in range(Nelx):
            e = ey * Nelx + ex
            X, Y = element_xy(ex, ey, Nelx, Nely, xLa, xLb, yLa, yLb, k)
            Z = Utiles[e]
            ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False,
                            rstride=1, cstride=1, color=num_color, alpha=alpha)
    # exact overlay (if provided)
    if EXtiles is not None:
        for ey in range(Nely):
            for ex in range(Nelx):
                e = ey * Nelx + ex
                X, Y = element_xy(ex, ey, Nelx, Nely, xLa, xLb, yLa, yLb, k)
                Z = EXtiles[e]
                ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False,
                                rstride=1, cstride=1, color=ex_color, alpha=alpha)
        # legend proxies
        handles = [Patch(facecolor=num_color, alpha=alpha, label="Numerical"),
                   Patch(facecolor=ex_color,  alpha=alpha, label="Exact")]
        ax.legend(handles=handles, loc="upper right")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u")
    ax.set_title(title)

def main():
    ap = argparse.ArgumentParser(description="Overlay numerical & exact DG2D surfaces from HDF5.")
    ap.add_argument("h5file", help="Path to HDF5 written by the solver")
    ap.add_argument("--nelx", type=int, help="Override Nelx")
    ap.add_argument("--nely", type=int, help="Override Nely")
    ap.add_argument("--alpha", type=float, default=0.75, help="Surface transparency [0..1]")
    ap.add_argument("--num-color", default="b", help="Color for numerical surface") # or use C0 is default matplotlib blue
    ap.add_argument("--ex-color",  default="y", help="Color for exact surface") #or use C8 is yellowish
    ap.add_argument("--elev", type=float, default=30.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--save", help="Save composite figure as PNG")
    args = ap.parse_args()

    with h5py.File(args.h5file, "r") as hf:
        # Domain
        G = hf["/Grid"]
        xLa = float(G.attrs["xLa"]); xLb = float(G.attrs["xLb"])
        yLa = float(G.attrs["yLa"]); yLb = float(G.attrs["yLb"])
        Lx, Ly = (xLb - xLa), (yLb - yLa)
        aspect = (Lx / Ly) if Ly != 0 else 1.0

        # Numerical
        NG = hf["/NumericalSolution"]
        Nel = int(read_attr(NG.attrs, "Nel"))
        k   = int(read_attr(NG.attrs, "k"))
        method = read_attr(NG.attrs, "method", "")
        u_h = np.array(NG["u_h"])
        k1 = k + 1
        assert u_h.size == Nel * k1 * k1, f"u_h length {u_h.size} != Nel*(k+1)^2={Nel*k1*k1}"

        # Try to use stored Nelx/Nely if present; else factor
        Nelx_attr = read_attr(NG.attrs, "Nelx")
        Nely_attr = read_attr(NG.attrs, "Nely")
        if args.nelx and args.nely:
            Nelx, Nely = args.nelx, args.nely
        elif Nelx_attr is not None and Nely_attr is not None:
            Nelx, Nely = int(Nelx_attr), int(Nely_attr)
        else:
            Nelx, Nely = best_factor_pair(Nel, aspect)

        Utiles = reshape_tiles(u_h, Nelx, Nely, k)

        # Exact (optional)
        EXtiles = None
        if "/ExactSolution/u_exact" in hf:
            u_ex = np.array(hf["/ExactSolution/u_exact"])
            if u_ex.size == u_h.size:
                EXtiles = reshape_tiles(u_ex, Nelx, Nely, k)
            else:
                print(f"[warn] exact length {u_ex.size} != numerical {u_h.size}; plotting numerical only.")

        # Composite plot
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=args.elev, azim=args.azim)

        title = f"{method}  k={k}  Nel={Nelx}×{Nely}  [{xLa},{xLb}]×[{yLa},{yLb}]"
        plot_both_surfaces(ax, Utiles, EXtiles, Nelx, Nely, xLa, xLb, yLa, yLb, k,
                           title, num_color=args.num_color, ex_color=args.ex_color, alpha=args.alpha)

        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
        plt.show()

if __name__ == "__main__":
    main()
