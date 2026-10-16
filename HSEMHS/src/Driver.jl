# Entry point (implementation doc §16.1).

const EXIT_CODES = Dict(converged => 0, stalled => 2, max_iter => 2, eos_failure => 3, invalid_input => 4)

"""
    run_config(path) -> exit code

Parse the config, dispatch on run.mode. Milestones not yet built report invalid_input.
"""
function run_config(path::AbstractString)
    ensure_mpi()
    status = converged
    try
        cfg = load_config(path)
        mode = cfg.run.mode
        if mode == "gate0"
            gate0_audit(cfg)
        elseif mode == "eos_table"
            build_eos_table(cfg)
        elseif mode == "column_hse"
            status = run_column_hse(cfg)
        elseif mode == "screening"
            status = run_screening(cfg)
        else
            invalid("run.mode = \"$mode\" is not implemented yet (available: gate0, eos_table, column_hse, screening)")
        end
    catch e
        e isa InvalidInput || rethrow()
        MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println(stderr, sprint(showerror, e))
        status = invalid_input
    end
    return EXIT_CODES[status]
end
