#!/usr/bin/env python3
import itertools
import subprocess

EXEC = "./testSolver"  # adjust path if needed
#EXEC = "./testSolver2"
write = "w"
methods = ["SIPG", "IIPG", "NIPG"]
ks = [1, 2, 3, 4, 5]
sizes = [2, 3, 4, 5, 6]  # Nel_x = Nel_y = size  ->  2x2 ... 6x6

for method, k, n in itertools.product(methods, ks, sizes):
    #cmd = [EXEC, method, str(k), str(n), str(n)]
    cmd = [EXEC, write, method, str(k), str(n), str(n)]
    print("Running:", " ".join(cmd))
    try:
        # Use capture_output=True if you want to collect stdout/stderr
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run failed (method={method}, k={k}, Nel={n}x{n}) with code {e.returncode}")
        # Uncomment to stop on first failure:
        # raise


