#!/bin/bash
# Step 3 of design/CODEX_REVIEW_solver_time_round6_2026-10-29.md: does better inner accuracy reduce total nonlinear
# work? The same μ0.3 vertical-balance full cube, the same column-HSE start and objective, max_steps outer steps with
#   base     the baseline (D2: Fourier, Eisenstat–Walker η ∈ [0.005, 0.1], cap 200)
#   blk      column blocks at the baseline forcing (η ∈ [0.005, 0.1], cap 200): the preconditioner alone
#   tight    column blocks at η = 1e-3, cap 3000 (diagnostic budget): inner accuracy reliably attained
#   base_t, blk_t, tight_t   the same from the state after base's steps (the tail: start=out/delivery/inner_base,
#            common_weights=1, i.e. the column-HSE start's objective, the one base minimises)
# run one after another on this node (comparable timings) by scripts/viewing/measure_steps.jl, which records per step
# the actual and model decrease, the cap and line-search scales, CG iterations and residuals, and seconds.
# usage: inner_accuracy.sh RANKS THREADS [STEPS] [VARIANT ...]   → out/delivery/inner_<variant>/
NR=$1; NT=$2; ST=${3:-10}; shift 3 || shift $#
IA_VARIANTS=${*:-base blk tight}
source ~/loadgcc.sh >/dev/null 2>&1
export PATH=/tmp/julia-1.13.1/bin:$HOME/.julia/bin:$PATH
cd /mn/stornext/d9/data/harshm/MHS_view/HSEMHS
IN=out/view/input_mu0.3_az0.nc
CUT=$(python3 -c "import math; print(-4 - math.log10(0.3))")       # vertical cut −4 on the line-of-sight axis
COMMON="max_steps=$ST stop=gn gn_tol=1e-3 logtau_cut=$CUT balance=vertical lm=gain forcing=ew"
for variant in $IA_VARIANTS; do
  T=""; [[ $variant == *_t ]] && T="start=out/delivery/inner_base common_weights=1"
  case ${variant%_t} in
    base)  X="precond=fourier cg_cap=200" ;;
    blk)   X="precond=block cg_cap=200" ;;
    tight) X="precond=block eta_min=1e-3 eta_max=1e-3 cg_cap=3000" ;;
  esac
  mkdir -p out/delivery/inner_$variant
  mpiexecjl --project=. -n $NR julia --project=. -t $NT scripts/viewing/measure_steps.jl $IN out/delivery/inner_$variant \
    0.3 0 inclined $COMMON $X $T > out/delivery/inner_$variant/steps.txt 2>&1
  echo "exit $?" >> out/delivery/inner_$variant/steps.txt
done
