#!/bin/bash
# Pin an MPI rank to one NUMA domain (cores and memory): numa_bind.sh <command…>. Rank r of a node uses domain
# (r mod NUMA_PER_NODE) (default: all domains of this node, one rank each), from MPICH's MPI_LOCALRANKID / PMI_RANK or OpenMPI's
# OMPI_COMM_WORLD_LOCAL_RANK.
n=${NUMA_PER_NODE:-$(lscpu | awk '/^NUMA node\(s\)/ {print $3}')}
r=${MPI_LOCALRANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_RANK:-0}}}
d=$(( r % n ))
exec numactl --cpunodebind=$d --membind=$d "$@"
