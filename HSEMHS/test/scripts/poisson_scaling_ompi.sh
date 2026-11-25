#!/bin/bash
# Multi-node strong scaling of poisson_scaling.jl over InfiniBand: OpenMPI 5.0.8 built against the nodes' UCX
# (~/packages/openmpi-5.0.8-ucx, no hcoll) and a Julia environment whose MPI.jl uses it (MPIPreferences system binary,
# ENV JENV, default /mn/stornext/d9/data/harshm/hsemhs_env_ompi, which develops this checkout). One rank per NUMA domain
# (numa_bind.sh). poisson_scaling_ompi.sh RANKS_PER_NODE THREADS host1 host2 ...  (ENV MODE, STEPS, POLISH, ADJ, PROFILE
# passed through). Each host needs /tmp/julia-1.12.3.
source ~/loadgcc.sh >/dev/null 2>&1
P=${OMPI:-$HOME/packages/openmpi-5.0.8-ucx}; JENV=${JENV:-/mn/stornext/d9/data/harshm/hsemhs_env_ompi}
export PATH=$P/bin:$PATH LD_LIBRARY_PATH=$P/lib        # not loadgcc's OpenMPI 5.0.5 (its libprrte, its libhcoll)
J=/tmp/julia-1.12.3/bin/julia; PPN=$1; T=$2; shift 2
cd "$(dirname "$0")/../.."
HOSTS=$(for h in "$@"; do printf "%s:%s," $h $PPN; done); N=$(( PPN * $# ))
export NUMA_PER_NODE=$PPN MODE=${MODE:-ltau} STEPS=${STEPS:-4} POLISH=${POLISH:-0} ADJ=${ADJ:-0} PROFILE=${PROFILE:-}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_0:1} UCX_WARN_UNUSED_ENV_VARS=n
exec mpiexec --prefix $P --host ${HOSTS%,} -n $N --bind-to none --mca pml ucx --mca btl ^tcp,uct \
     -x PATH -x LD_LIBRARY_PATH -x NUMA_PER_NODE -x MODE -x STEPS -x POLISH -x ADJ -x PROFILE -x UCX_NET_DEVICES \
     -x UCX_WARN_UNUSED_ENV_VARS \
     test/scripts/numa_bind.sh $J --startup-file=no --project=$JENV -t $T \
     -e 'using LinearAlgebra; BLAS.set_num_threads(1); include("test/scripts/poisson_scaling.jl")'
