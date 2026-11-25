#!/bin/bash
# One group of the adjoint tests on this node (called by adjoint_parallel.sh): adjoint_node.sh serial GROUP CASES THREADS
# or adjoint_node.sh mpi RANKS CASES THREADS
source ~/loadgcc.sh >/dev/null 2>&1
rsync -a /mn/stornext/astro/local/julia/julia-1.12.3/ /tmp/julia-1.12.3/
J=/tmp/julia-1.12.3/bin/julia
cd "$(dirname "$0")/../.."
export ADJ_CASES="$3"
if [ "$1" = serial ]; then
  ADJ_GROUP="$2" exec $J --startup-file=no --project=. -t "$4" test/run_adjoint.jl
else
  MPIEXEC=$($J --startup-file=no --project=. -e 'using MPI; print(MPI.mpiexec().exec[1])')
  exec $MPIEXEC -n "$2" $J --startup-file=no --project=. -t "$4" test/mpi/mpi_adjoint.jl
fi
