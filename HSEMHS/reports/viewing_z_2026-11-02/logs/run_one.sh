#!/bin/bash
# usage: run_one.sh NAME RANKS THREADS
source ~/loadgcc.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.13.1/bin:$HOME/.julia/bin:$PATH
cd /mn/stornext/d9/data/harshm/MHS_solvertime/HSEMHS
start=$(date +%s); echo "=== $1 on $(hostname) $(date)"
mpiexecjl --project=. -n $2 julia --project=. -t $3 scripts/run.jl config/modes/zview_$1.toml
echo "EXIT $? wall $(( $(date +%s) - start )) s"
