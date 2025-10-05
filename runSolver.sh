#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
#BIN="./testSolver_sp"
BIN="./testSolver_sp_ldlt"

pi=3.141592653589793

WRITE="w"                 # "w" to write+plot, anything else = just run
METHOD="NIPG"
#METHOD="IIPG"
#METHOD="NIPG"
#METHOD="NIPG0"             # SIPG | IIPG | NIPG | NIPG0
K=3
#SIGMA0=$((2*K*K+1))
SIGMA0=$((K*K+1))
NELX=16
NELY=16
XLA=0
#XLB=1
XLB=$(echo "scale=6; 2*$pi" | bc -l)
YLA=0
#YLB=1
YLB=$(echo "scale=6; 2*$pi" | bc -l)
BC_SPEC="L=D,R=D,B=N,T=N"
#CASE_JSON="cases/gaussian.json"
CASE_JSON="cases/testcase1.json"
#CASE_JSON="cases/testcase0.json"
#CASE_JSON="cases/helmholtz.json"

# If your C++ writes into a folder, set it here; leave empty if not.
OUTDIR="./data"                 # e.g. "data"

# ---- run solver ----
echo ">>> $BIN $WRITE $METHOD $K $SIGMA0 $NELX $NELY $XLA $XLB $YLA $YLB \"$BC_SPEC\" $CASE_JSON"
"$BIN" "$WRITE" "$METHOD" "$K" "$SIGMA0" "$NELX" "$NELY" "$XLA" "$XLB" "$YLA" "$YLB" "$BC_SPEC" "$CASE_JSON"

# ---- plot only if WRITE == "W" ----
if [[ "$WRITE" == "w" ]]; then
  case_name="$(basename "$CASE_JSON")"; case_name="${case_name%.*}"
  nel=$(( NELX * NELY ))

  # Effective sigma0 for the **filename** (solver forces these internally)
  SIGMA0_EFF="$SIGMA0"
  case "$METHOD" in
    NIPG)  SIGMA0_EFF="1" ;;
    NIPG0) SIGMA0_EFF="0" ;;
  esac

  fname="DG2DRec_${case_name}_method=${METHOD}_Nel=${nel}_k=${K}_sigma0=${SIGMA0_EFF}.h5"
  [[ -n "$OUTDIR" ]] && h5path="${OUTDIR%/}/$fname" || h5path="$fname"

  echo ">>> python3 ./scripts/plotSolution.py $h5path"
  python3 ./scripts/plotSolution.py "$h5path"
fi

