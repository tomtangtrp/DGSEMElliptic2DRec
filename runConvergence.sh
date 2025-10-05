#!/usr/bin/env bash
set -euo pipefail

# ---- method arg (required; only 4 allowed) ----
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <testcase> {SIPG|IIPG|NIPG|NIPG0}"
  echo " <testcase> should be in ./cases/testcase0.json"
  exit 1
fi

CASE_NAME="$1"
METHOD="$2"
case "$METHOD" in
  SIPG|IIPG|NIPG|NIPG0) ;;
  *) echo "ERROR: METHOD must be one of SIPG|IIPG|NIPG|NIPG0"; exit 1 ;;
esac

# ---- config (match your solver) ----
pi=3.141592653589793
BIN="./testSolver_sp"
WRITE="w"                      # must be 'w' so the solver writes HDF5
XLA=0; 
XLB=1; 
#XLB=$(echo "scale=6; 2*$pi" | bc -l)
YLA=0; 
#YLB=1
YLB=$(echo "scale=6; 2*$pi" | bc -l)
BC_SPEC="L=D,R=D,B=N,T=N"
CASE_JSON="cases/${CASE_NAME}.json"
#CASE_JSON="cases/helmholtz.json"
#CASE_JSON="cases/testcase1.json"

# Nel targets required by plotConvergence.py: 4,9,16,25,36 => Nelx=Nely=2..6
#N1D_LIST=(2 3 4 5 6)
N1D_LIST=(2 4 8 16 32)

echo "=== Convergence sweep: METHOD=${METHOD}, case=${CASE_JSON} ==="
for K in 1 2 3 4 5; do
  # Simple sigma0 choice; NIPG/NIPG0 will override inside solver anyway
  SIGMA0=$(( 2*K*K + 1 ))
  for N1D in "${N1D_LIST[@]}"; do
    NELX=$N1D
    NELY=$N1D
    echo ">>> k=${K}, Nelx=Nely=${N1D} (Nel=$((NELX*NELY)))"
    "$BIN" "$WRITE" "$METHOD" "$K" "$SIGMA0" "$NELX" "$NELY" "$XLA" "$XLB" "$YLA" "$YLB" "$BC_SPEC" "$CASE_JSON"
  done
done

echo "=== Done. Files should be in ./data ==="

# --- plot convergence figures ---
if [[ -f ./scripts/plotConvergence.py ]]; then
  echo "=== Plotting convergence: case=${CASE_JSON}, METHOD=${METHOD} ==="
  python3 ./scripts/plotConvergence.py "${CASE_JSON}" "${METHOD}"
else
  echo "WARNING: ./scripts/plotConvergence.py not found; skipping plots."
fi
