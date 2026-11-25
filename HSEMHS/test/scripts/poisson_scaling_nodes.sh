#!/bin/bash
# Multi-node strong scaling of poisson_scaling.jl over MPI.jl's MPICH, one rank per NUMA domain (numa_bind.sh):
# poisson_scaling_nodes.sh RANKS_PER_NODE THREADS host1 host2 ...   (ENV MODE, STEPS, POLISH, ADJ passed through;
# Zen 3 hyades nodes; IFACE: the network interface, default the InfiniBand ibp65s0, not the 1 Gb/s eno8303).
# Each host needs /tmp/julia-1.12.3.
source ~/loadgcc.sh >/dev/null 2>&1
J=/tmp/julia-1.12.3/bin/julia; PPN=$1; T=$2; shift 2
cd "$(dirname "$0")/../.."
HOSTS=$(IFS=,; echo "$*"); N=$(( PPN * $# ))
MPIEXEC=$($J --startup-file=no --project=. -e 'using MPI; print(MPI.mpiexec().exec[1])')
exec $MPIEXEC -hosts "$HOSTS" -ppn $PPN -n $N -iface ${IFACE:-ibp65s0} -genv FI_PROVIDER tcp -genv FI_TCP_IFACE ${IFACE:-ibp65s0} \
     -genv NUMA_PER_NODE $PPN -genv MODE "${MODE:-ltau}" -genv STEPS "${STEPS:-4}" -genv POLISH "${POLISH:-0}" -genv ADJ "${ADJ:-0}" -genv PROFILE "${PROFILE:-}" \
     test/scripts/numa_bind.sh $J --startup-file=no --project=. -t $T \
     -e 'using LinearAlgebra; BLAS.set_num_threads(1); include("test/scripts/poisson_scaling.jl")'
