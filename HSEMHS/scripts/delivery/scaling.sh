#!/bin/bash
# Step 2 of design/CODEX_REVIEW_solver_time_round6_2026-10-29.md: hardware scaling of mhs-poisson-ltau at fixed work.
# The same full cube, column-HSE start and objective; STEPS outer steps, each CG run to exactly CGIT iterations
# (forcing fixed, cg_tol 1e-12 is not reached; stop = phi cannot trigger in so few steps), so every layout does the same
# arithmetic and only the hardware changes. Through the production CLI (scripts/delivery/run_case.sh).
# usage: scaling.sh MU AZ INPUT "NODE1 NODE2 ..." "NNODES:RANKS_PER_NODE:THREADS ..." [STEPS] [CGIT] [PRECOND]
#   e.g. scaling.sh 0.3 0 out/view/input_mu0.3_az0.nc "hyades8 hyades12 hyades15 hyades16" "1:1:32 2:1:32 4:1:32"
MU=$1; AZ=$2; IN=$3; NODES=($4); LAYOUTS=$5; ST=${6:-3}; CG=${7:-100}; PC=${8:-fourier}
cd /mn/stornext/d9/data/harshm/MHS_view/HSEMHS
for L in $LAYOUTS; do
  IFS=: read NN RPN NT <<< "$L"
  NAME=scale_mu${MU}_${PC}_${NN}n${RPN}r${NT}t
  HF=out/delivery/$NAME.hosts; : > $HF
  for ((k = 0; k < NN; k++)); do echo "${NODES[$k]}:$RPN" >> $HF; done
  scripts/delivery/run_case.sh $NAME mhs-poisson-ltau $IN $MU $AZ $((NN * RPN)) $NT $HF \
    max_steps=$ST stop='"phi"' forcing='"fixed"' cg_tol=1.0e-12 cg_maxit=$CG precond=\"$PC\"
done
