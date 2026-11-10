#!/bin/bash
# diag_pert.sh THREADS TAG...: force_diag.jl of colhse_pert_TAG and mhs_pert_TAG against ../IO/tau252/perturbed/input_TAG.nc
export PATH=/tmp/julia-1.13.1/bin:$PATH
cd /mn/stornext/d9/data/harshm/MHS/HSEMHS
D=out/paper_252_2026-11-10; T=$1; shift
for t in "$@"; do for k in colhse mhs; do
  julia --project=. -t $T scripts/ltaucut/force_diag.jl ../IO/tau252/perturbed/input_$t.nc $D/${k}_pert_$t.h5 $D/diag_${k}_pert_$t.h5 > $D/diag_${k}_pert_$t.log 2>&1; echo "${k}_pert_$t EXIT $?"
done; done
