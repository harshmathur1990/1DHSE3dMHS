#!/bin/bash
# hyades3: bash -l test/scripts_adjoint_run.sh [serial|mpi N]
source ~/loadgcc.sh >/dev/null 2>&1
cd /mn/stornext/d9/data/harshm/HSEMHS/HSEMHS
J=/tmp/julia-1.12.3/bin/julia
if [ "${1:-serial}" = serial ]; then
  exec $J --startup-file=no --project=. -t 8 test/run_adjoint.jl
else
  MPIEXEC=$($J --startup-file=no --project=. -e 'using MPI; print(MPI.mpiexec().exec[1])')
  exec $MPIEXEC -n $2 $J --startup-file=no --project=. -t 2 test/mpi/mpi_adjoint.jl
fi
