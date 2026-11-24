#!/bin/bash
# Run the equilibrium-adjoint tests split over hyades nodes at once (one group per node, threads inside, the MPI checks
# on their own nodes): bash test/scripts/adjoint_parallel.sh [GROUP ...]   (default: every group below)
# Groups are NAME=HOST:KIND:ARG, KIND serial (ARG = ADJ_GROUP/ADJ_CASES) or mpi (ARG = ranks/ADJ_CASES).
# Logs: test/logs/adjoint/NAME.log; the summary lists each group's Test Summary line and exit status.
set -u
here=$(cd "$(dirname "$0")/../.." && pwd)              # HSEMHS/HSEMHS
logs=$here/test/logs/adjoint; mkdir -p "$logs"
ALL=(
  "units=hyades5:serial:units/.*"
  "ltau=hyades9:serial:cases/^column-hse-ltau"
  "logm=hyades10:serial:cases/^column-hse-logm"
  "z=hyades12:serial:cases/^column-hse-z"
  "pz=hyades13:serial:cases/^mhs-poisson-z"
  "pltau=hyades14:serial:cases/^mhs-poisson-ltau"
  "plogm=hyades15:serial:cases/^mhs-poisson-logm"
  "mpi2=hyades16:mpi:2/.*"
  "mpi3=hyades2:mpi:3/.*"
)
groups=("$@"); [ ${#groups[@]} -eq 0 ] && groups=("${ALL[@]}")
pids=(); names=()
for g in "${groups[@]}"; do
  name=${g%%=*}; rest=${g#*=}; host=${rest%%:*}; rest=${rest#*:}; kind=${rest%%:*}; arg=${rest#*:}
  a=${arg%%/*}; cases=${arg#*/}
  if [ "$kind" = serial ]; then th=16; else th=4; fi
  ssh -o BatchMode=yes "$host" "bash -l $(printf %q "$here/test/scripts/adjoint_node.sh") $kind $a $(printf %q "$cases") $th" \
      > "$logs/$name.log" 2>&1 &
  pids+=($!); names+=("$name@$host")
done
fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  line=$(grep -E "^(equilibrium adjoint|Test Summary)" -A1 "$logs/${names[$i]%@*}.log" | grep -E "equilibrium adjoint" | tail -1)
  printf '%-14s exit %d  %s\n' "${names[$i]}" "$rc" "$line"
  [ $rc -ne 0 ] && fail=1
done
exit $fail
