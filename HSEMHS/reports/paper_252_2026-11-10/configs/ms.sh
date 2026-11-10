#!/bin/bash
# ms.sh CASE NRANKS THREADS: scripts/viewing/measure_steps.jl on ../IO/tau252/input_CASE.nc (μ = 1), release settings, every step saved to ms_CASE/
export PATH=/tmp/julia-1.13.1/bin:$HOME/.julia/bin:$PATH
cd /mn/stornext/d9/data/harshm/MHS/HSEMHS
D=out/paper_252_2026-11-10; mkdir -p $D/ms_$1
echo "host $(hostname) commit $(git rev-parse --short HEAD) ranks $2 threads $3 start $(date -Is)" > $D/ms_$1.log
/usr/bin/time -f "WALL %e s, maxrss/rank %M kB" mpiexecjl --project=. -n $2 mhs_rank.sh julia --project=. -t $3 \
  scripts/viewing/measure_steps.jl ../IO/tau252/input_$1.nc $D/ms_$1 1.0 0.0 plane_parallel \
  stop=gn lm=gain forcing=ew precond=switch cg_cap=200 gn_tol=1e-3 max_steps=100 logtau_cut=-4 >> $D/ms_$1.log 2>&1
echo "EXIT $? end $(date -Is)" >> $D/ms_$1.log
