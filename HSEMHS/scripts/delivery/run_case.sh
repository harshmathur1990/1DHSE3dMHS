#!/bin/bash
# Delivery matrix (design/CODEX_REVIEW_solver_time_round6_2026-10-29.md, steps 1–2): one case through the production
# CLI (scripts/run.jl), with per-rank peak memory and wall time from /usr/bin/time.
#
# usage: run_case.sh NAME MODE INPUT MU AZIMUTH RANKS THREADS [HOSTFILE|-] [KEY=VALUE ...]
#   MODE      column-hse-ltau | mhs-poisson-ltau
#   INPUT     τ-grid input (out/view/input_mu*_az*.nc), geometry inclined
#   HOSTFILE  MPICH host file for multi-node runs (- or absent: this node)
#   KEY=VALUE extra [poisson] keys (TOML values, e.g. lm='"gain"' max_steps=5)
# Writes out/delivery/NAME.toml, NAME.log, NAME.h5 and NAME.time.<host>.<pid> (peak RSS KB, elapsed s per rank).
# The baseline poisson settings (D2) are written first; KEY=VALUE pairs given here are appended and override them.
# The log ends with "exit N": 0 converged, 2 valid but not converged (max_iter/stalled; the output is kept and its
# attributes say so), 3 invalid result (nothing written), 4 invalid input. This script records N and does not stop on it.
NAME=$1; MODE=$2; INPUT=$3; MU=$4; AZ=$5; NR=$6; NT=$7; HF=${8:--}; shift 8 || shift $#
source ~/loadgcc.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.13.1/bin:$HOME/.julia/bin:$PATH
cd /mn/stornext/d9/data/harshm/MHS_view/HSEMHS
mkdir -p out/delivery
C=out/delivery/$NAME.toml
GEOM=inclined; [ "$MU" = "1" ] || [ "$MU" = "1.0" ] && GEOM=plane_parallel
{
  echo "[run]"; echo "mode = \"$MODE\""; echo "output = \"out/delivery/$NAME.h5\""; echo "overwrite = true"
  echo "[input]"; echo "file = \"$INPUT\""
  echo "[physics]"; echo "g = 274.0"
  echo "[eos]"; echo "table_file = \"cache/eos_table_fullz.h5\""
  echo "[tau_grid]"; echo "align = \"ptot\""
  echo "[view]"; echo "mu = $MU"; echo "azimuth_deg = $AZ"; echo "geometry = \"$GEOM\""
  if [ "$MODE" = "mhs-poisson-ltau" ]; then
    echo "[poisson]"
    declare -A P=([logtau_cut]=-4.0 [max_steps]=25 [stop]='"gn"' [gn_tol]=1.0e-3 [lm]='"gain"' [forcing]='"ew"' [cg_cap]=200)
    for kv in "$@"; do P[${kv%%=*}]=${kv#*=}; done
    for k in "${!P[@]}"; do echo "$k = ${P[$k]}"; done
  fi
} > $C
W="/usr/bin/time -f %M\ %e -o out/delivery/$NAME.time.\$(hostname -s).\$\$"
if [ "$HF" = "-" ]; then
  mpiexecjl --project=. -n $NR bash -c "$W julia --project=. -t $NT scripts/run.jl $C" > out/delivery/$NAME.log 2>&1; rc=$?
else
  mpiexecjl --project=. -launcher ssh -launcher-exec $HOME/.julia/bin/ssh_bash_launcher -f $HF -n $NR \
    mhs_rank.sh bash -c "$W julia --project=. -t $NT scripts/run.jl $C" > out/delivery/$NAME.log 2>&1; rc=$?
fi
echo "exit $rc" >> out/delivery/$NAME.log
