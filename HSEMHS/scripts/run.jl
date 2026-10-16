# Usage: [mpiexec -n R] julia --project=. -t T scripts/run.jl config.toml
using HSEMHS
using MPI

length(ARGS) == 1 || (println(stderr, "usage: julia --project=. scripts/run.jl config.toml"); exit(4))
code = HSEMHS.run_config(ARGS[1])
MPI.Finalized() || MPI.Finalize()
exit(code)
