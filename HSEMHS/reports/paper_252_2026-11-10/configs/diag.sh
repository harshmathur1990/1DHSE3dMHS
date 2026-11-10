#!/bin/bash
# diag.sh THREADS STATE [STATE ...]: force balance (scripts/ltaucut/force_diag.jl) of STATE.h5 (a run, or truth:CASE) against its input
export PATH=/tmp/julia-1.13.1/bin:$PATH
cd /mn/stornext/d9/data/harshm/MHS/HSEMHS
D=out/paper_252_2026-11-10; T=$1; shift
for s in "$@"; do
  case $s in truth:*) c=${s#truth:}; st=../IO/tau252/truth_$c.h5; out=$D/diag_truth_$c.h5;;
             *) c=$(sed -E 's/^[a-z0-9]+_//' <<< $s); st=$D/$s.h5; out=$D/diag_$s.h5;; esac
  julia --project=. -t $T scripts/ltaucut/force_diag.jl ../IO/tau252/input_$c.nc $st $out > ${out%.h5}.log 2>&1; echo "$s EXIT $?"
done
