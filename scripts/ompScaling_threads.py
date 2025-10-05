#!/usr/bin/env python3
# OpenMP scaling harness for DG2DRec solver (matches 13-arg CLI).
# Keeps the original environment preset logic; only adapts to the new exec inputs.
import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from statistics import mean, stdev
import matplotlib.pyplot as plt
import pathlib

# ---------- helpers (unchanged behavior) ----------
def detect_omp_runtime(exe_path: str) -> str:
    """Return 'gomp', 'iomp', or 'unknown' by scanning ldd once."""
    try:
        out = subprocess.check_output(["ldd", exe_path], text=True)
    except Exception:
        return "unknown"
    if "libgomp" in out:
        return "gomp"
    if "libiomp5" in out:
        return "iomp"
    return "unknown"

def build_env(preset: str, threads: int, base_env: dict, proc_bind: str | None) -> dict:
    env = base_env.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    if proc_bind and proc_bind.lower() != "inherit":
        env["OMP_PROC_BIND"] = proc_bind

    if preset == "inherit":
        return env

    if preset == "minimal":
        env.pop("OMP_DYNAMIC", None)
        for k in ("OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.pop(k, None)
        for k in ("MKL_NUM_THREADS", "MKL_DYNAMIC", "MKL_THREADING_LAYER"):
            env.pop(k, None)
        return env

    if preset == "mkl-seq":
        env["MKL_NUM_THREADS"] = "1"
        env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        env.pop("MKL_DYNAMIC", None)
        return env

    if preset == "mkl-gnu":
        env["MKL_THREADING_LAYER"] = "GNU"
        env.pop("MKL_DYNAMIC", None)
        if threads > 1:
            env["MKL_NUM_THREADS"] = "1"
        else:
            env.pop("MKL_NUM_THREADS", None)
        return env

    return env  # fallback

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="OpenMP scaling for DG2DRec (serial solver CLI compatible)")
    parser.add_argument("--exe", default="./testSolver_sp",
                        help="Path to executable (default: ./testSolver_sp)")
    parser.add_argument("--min", type=int, default=1, help="Minimum threads")
    parser.add_argument("--max", type=int, help="Maximum threads (default: all logical cores)")
    parser.add_argument("--trials", type=int, default=1, help="Trials per thread count")
    parser.add_argument("--start", type=int, default=2,
                        help="1: test [1,2,4,6,...], else: [2,4,6,...]")

    # Environment control (as before)
    parser.add_argument("--env-preset", choices=["inherit", "minimal", "mkl-seq", "mkl-gnu"],
                        default="inherit", help="OMP/MKL preset")
    parser.add_argument("--proc-bind", choices=["inherit", "close", "spread", "master"],
                        default="inherit", help="Set OMP_PROC_BIND if not 'inherit'")
    parser.add_argument("--preflight-ldd", action="store_true",
                        help="Print detected OpenMP runtime via ldd")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue sweep and record NaN if a run fails")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")

    # Everything after '--' is passed to the solver
    parser.add_argument("solver_args", nargs=argparse.REMAINDER,
                        help=("Solver args:\n"
                              "{w|nw} METHOD k sigma0 Nel_x Nel_y xLa xLb yLa yLb BC_SPEC CASE_JSON"))
    args = parser.parse_args()

    # Validate solver args against the new 13-arg interface (program + 12 args)
    if not args.solver_args:
        parser.error("No solver arguments provided.\nExample:\n"
                     "  w SIPG 3 25 8 8 0 1 0 1 L=D,R=D,B=D,T=D cases/testcase1.json")

    # Strip a leading '--' if user separated with it
    if args.solver_args and args.solver_args[0] == '--':
        args.solver_args = args.solver_args[1:]

    if len(args.solver_args) != 12:
        parser.error(f"Expected 12 solver args, got {len(args.solver_args)}.\n"
                     "Format: {w|nw} METHOD k sigma0 Nel_x Nel_y xLa xLb yLa yLb BC_SPEC CASE_JSON")

    mode_flag = args.solver_args[0]
    if mode_flag.lower() not in ("w", "nw"):
        parser.error(f"First solver arg must be 'w' or 'nw', got: {mode_flag}")

    exe_path = args.exe
    if "/" in exe_path:
        if not (os.path.exists(exe_path) and os.access(exe_path, os.X_OK)):
            raise FileNotFoundError(f"Executable not found or not executable: {exe_path}")
    else:
        if not shutil.which(exe_path):
            raise FileNotFoundError(f"Executable not found on PATH: {exe_path}")

    detected_max = multiprocessing.cpu_count()
    max_threads = args.max if args.max else detected_max
    print(f"Detected {detected_max} logical cores; scaling from {args.min} to {max_threads} threads")

    if args.preflight_ldd:
        rt = detect_omp_runtime(exe_path)
        print(f"OpenMP runtime (ldd): {rt}")

    # Prepare thread counts
    if args.start == 1:
        thread_counts = [1, 2] + list(range(4, max_threads + 1, 2))
    else:
        start = max(2, args.min)
        thread_counts = list(range(start, max_threads + 1, 2))

    print(f"Testing thread counts: {thread_counts}")
    results = []

    # For titles/filenames
    METHOD   = args.solver_args[1]
    K        = int(args.solver_args[2])
    NELX     = int(args.solver_args[4])  # index 4 -> Nel_x (0-based: 0..11)
    NELY     = int(args.solver_args[5])  # index 5 -> Nel_y
    NEL      = NELX * NELY
    CASE_JSON = args.solver_args[-1]
    case_name = pathlib.Path(CASE_JSON).stem

    for nthreads in thread_counts:
        times = []
        for t in range(args.trials):
            env = build_env(args.env_preset, nthreads, os.environ,
                            None if args.proc_bind == "inherit" else args.proc_bind)

            cmd = [exe_path] + args.solver_args
            print(f"\nThreads={nthreads}, trial {t+1}/{args.trials}")
            print("CMD:", " ".join(cmd))
            log_keys = [k for k in env if k.startswith(("OMP", "MKL"))]
            print("ENV OMP/MKL:", {k: env[k] for k in sorted(log_keys)})

            start = time.perf_counter()
            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                rc = e.returncode
                print(f"[ERROR] Solver crashed (return code {rc}):", " ".join(e.cmd))
                if args.continue_on_error:
                    times.append(float("nan"))
                    continue
                else:
                    sys.exit(1)
            elapsed = time.perf_counter() - start
            print(f"Wall time: {elapsed:.6f} s")
            times.append(elapsed)

        good = [x for x in times if x == x]  # filter NaNs
        if good:
            mu = mean(good)
            sigma = stdev(good) if len(good) > 1 else 0.0
        else:
            mu, sigma = float("nan"), float("nan")
        results.append((nthreads, mu, sigma))

    # ----- save CSV / plot -----
    csv_name = f"walltime_vs_threads_{case_name}_method={METHOD}_Nel={NEL}_k={K}.csv"
    with open(csv_name, "w") as f:
        f.write("threads,mean_wall_time,stddev_wall_time\n")
        for n, mu, sigma in results:
            mu_str = f"{mu:.6f}" if mu == mu else "nan"
            sigma_str = f"{sigma:.6f}" if sigma == sigma else "nan"
            f.write(f"{n},{mu_str},{sigma_str}\n")
    print(f"Wrote {csv_name}")

    threads = [r[0] for r in results]
    means   = [r[1] for r in results]
    stds    = [r[2] for r in results]
    #plt.errorbar(threads, means, yerr=stds, fmt="o-", capsize=3)
    plt.loglog(threads, means, "--o")
    plt.xlabel("Number of CPU threads")
    plt.ylabel("Wall time (seconds)")
    plt.title(
        f"OMP scaling: case={case_name}, method={METHOD}, k={K}\n"
        f"Nel_x={NELX}, Nel_y={NELY}  |  mode={mode_flag}"
    )
    plt.grid(True, alpha=0.3)
    png_name = f"walltime_vs_threads_{case_name}_method={METHOD}_Nel={NEL}_k={K}.png"
    plt.savefig(png_name, dpi=150, bbox_inches="tight")
    print(f"Wrote {png_name}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()

