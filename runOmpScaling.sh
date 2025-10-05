#!/usr/bin/env bash
set -euo pipefail

# ---------- Solver binary ----------
EXE="./testSolver_sp"
#EXE="./testSolver_sp_mkleigen"

# ---------- Solver args (13-arg CLI expects 12 args after exe) ----------
WRITE="nw"                 # w | nw
METHOD="SIPG"             # SIPG | IIPG | NIPG | NIPG0
K=3
#SIGMA0=19
SIGMA0=$((K*K+1))
NELX=8
NELY=8
XLA=0
XLB=1
YLA=0
YLB=1
PI=3.141592653589793
#YLB=$(awk -v pi="$PI" 'BEGIN{printf "%.17g", 2*pi}')

BC_SPEC="L=D,R=D,B=D,T=D"
CASE_JSON="cases/testcase1.json"

# ---------- OMP scaling options ----------
TRIALS=3
ENV_PRESET="minimal"      # inherit | minimal | mkl-seq | mkl-gnu
PROC_BIND="close"         # inherit | close | spread | master
START_SERIES=1            # 1 => [1,2,4,6,...], else [2,4,6,...]
MIN_THREADS=1
MAX_THREADS=""            # leave empty to auto-detect
PREFLIGHT_LDD=1           # 1 to print OpenMP runtime via ldd
CONTINUE_ON_ERROR=0
SHOW_PLOT=0

# ---------- Build command ----------
ARGS=( "$WRITE" "$METHOD" "$K" "$SIGMA0" "$NELX" "$NELY" "$XLA" "$XLB" "$YLA" "$YLB" "$BC_SPEC" "$CASE_JSON" )

CMD=( python3 ./scripts/ompScaling.py
  --exe "$EXE"
  --trials "$TRIALS"
  --env-preset "$ENV_PRESET"
  --proc-bind "$PROC_BIND"
  --start "$START_SERIES"
  --min "$MIN_THREADS"
)

# Optional flags
if [[ -n "${MAX_THREADS}" ]]; then
  CMD+=( --max "$MAX_THREADS" )
fi
if [[ "$PREFLIGHT_LDD" == "1" ]]; then
  CMD+=( --preflight-ldd )
fi
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  CMD+=( --continue-on-error )
fi
if [[ "$SHOW_PLOT" == "1" ]]; then
  CMD+=( --show )
fi

# Single separator before the 12 solver args
CMD+=( -- "${ARGS[@]}" )

echo ">>> ${CMD[*]}"
exec "${CMD[@]}"

