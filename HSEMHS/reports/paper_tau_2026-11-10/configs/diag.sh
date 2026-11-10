#!/bin/bash
# diag.sh RUN THREADS: force balance of RUN.h5 (scripts/ltaucut/force_diag.jl) → diag_RUN.h5
export PATH=/tmp/julia-1.13.1/bin:$PATH
cd /mn/stornext/d9/data/harshm/MHS/HSEMHS
D=out/paper_tau_2026-11-10
julia --project=. -t $2 scripts/ltaucut/force_diag.jl $D/input-ltau-xy.nc $D/$1.h5 $D/diag_$1.h5 > $D/diag_$1.log 2>&1
echo "EXIT $?" >> $D/diag_$1.log
