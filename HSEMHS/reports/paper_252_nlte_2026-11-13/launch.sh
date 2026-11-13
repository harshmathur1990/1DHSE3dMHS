#!/bin/bash -l
# launch.sh NRANKS THREADS NAME [NAME ...]: run.jl with NAME.toml, one after the other (branch nlte-eos worktree)
export PATH=/tmp/julia-1.13.1/bin:$HOME/.julia/bin:$PATH
cd /mn/stornext/d9/data/harshm/MHS_nlte/HSEMHS
D=out/paper_252_nlte_2026-11-13
R=$1; T=$2; shift 2
for n in "$@"; do
  echo "host $(hostname) commit $(git rev-parse --short HEAD)$(git diff --quiet || echo +dirty) ranks $R threads $T start $(date -Is)" > $D/$n.log
  /usr/bin/time -f "WALL %e s, maxrss/rank %M kB" mpiexecjl --project=. -n $R mhs_rank.sh julia --project=. -t $T \
    scripts/run.jl $D/$n.toml >> $D/$n.log 2>&1
  echo "EXIT $? end $(date -Is)" >> $D/$n.log
done
