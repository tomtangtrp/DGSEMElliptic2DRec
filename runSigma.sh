# example
python3 ./scripts/scanSigma.py \
  --exe ./testSolver_sp \
  --method SIPG \
  -k 3 \
  --nelx 4 --nely 4 \
  --bc "L=D,R=D,B=D,T=D" \
  --case ./cases/testcase0.json \
  --write nw

