# Usage: [mpiexec -n R] julia --project=. -t T scripts/run.jl config.toml
#
# run.mode: column-hse-z | column-hse-ltau | mhs-poisson-z | mhs-poisson-ltau | full-mhs-z (all MPI + threads),
# plus gate0, eos_table, screening. full-mhs-z runs the M5 joint Gauss–Newton driver (scripts/m5_bifrost/m5h_fullres.jl)
# with the [full_mhs] settings (defaults = the M5i run: uniform 5 G σ_B, no trust radius, Φ stop); run.output is its
# output directory.
using HSEMHS
using MPI
using LinearAlgebra
BLAS.set_num_threads(1)      # standalone default (formerly set in HSEMHS.__init__); the τ driver raises it around its solve

length(ARGS) == 1 || (println(stderr, "usage: julia --project=. scripts/run.jl config.toml"); exit(4))
HSEMHS.ensure_mpi()
cfg = try
    HSEMHS.load_config(ARGS[1])
catch e
    e isa HSEMHS.InvalidInput || rethrow()
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println(stderr, sprint(showerror, e))
    MPI.Finalized() || MPI.Finalize()
    exit(4)
end
code = if cfg.run.mode == "full-mhs-z"
    F = cfg.full_mhs
    ENV["MHS_INPUT"] = abspath(cfg.input.file)
    ENV["MHS_EOS_TABLE"] = abspath(cfg.eos.table_file)
    ENV["MHS_SIGMA_REL"] = string(F.sigma_rel)
    ENV["MHS_TRUST"] = F.trust_region ? "on" : "off"
    ENV["MHS_STOP"] = F.stop
    ENV["MHS_PHI_RTOL"] = string(F.phi_rtol)
    include(joinpath(@__DIR__, "m5_bifrost", "m5h_fullres.jl"))
    Base.invokelatest(main, [cfg.run.output, F.run, string(F.wall_hours), string(F.ckpt_hours), string(F.init_iter), F.restart])
    0
else
    HSEMHS.run_config(ARGS[1])
end
MPI.Finalized() || MPI.Finalize()
exit(code)
