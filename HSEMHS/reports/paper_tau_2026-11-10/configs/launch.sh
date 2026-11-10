#!/bin/bash
# launch.sh NAME NRANKS THREADS            run.jl with NAME.toml
# launch.sh ms100 NRANKS THREADS           measure_steps.jl, release settings, 100 steps, every step saved to ms100/
export PATH=/tmp/julia-1.13.1/bin:$HOME/.julia/bin:$PATH
cd /mn/stornext/d9/data/harshm/MHS/HSEMHS
D=out/paper_tau_2026-11-10
echo "host $(hostname) commit $(git rev-parse --short HEAD) ranks $2 threads $3 start $(date -Is)" > $D/$1.log
if [ "$1" = ms100 ]; then
  mkdir -p $D/ms100
  /usr/bin/time -f "WALL %e s, maxrss/rank %M kB" mpiexecjl --project=. -n $2 mhs_rank.sh julia --project=. -t $3 \
    scripts/viewing/measure_steps.jl $D/input-ltau-xy.nc $D/ms100 1.0 0.0 plane_parallel \
    stop=gn lm=gain forcing=ew precond=switch cg_cap=200 gn_tol=1e-3 max_steps=100 logtau_cut=-4 >> $D/$1.log 2>&1
else
  /usr/bin/time -f "WALL %e s, maxrss/rank %M kB" mpiexecjl --project=. -n $2 mhs_rank.sh julia --project=. -t $3 \
    scripts/run.jl $D/$1.toml >> $D/$1.log 2>&1
fi
echo "EXIT $? end $(date -Is)" >> $D/$1.log
