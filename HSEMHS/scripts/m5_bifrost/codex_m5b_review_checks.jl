# Independent review checks for CODEX_REVIEW_M5_FOLLOWUP.md.
# Run from HSEMHS: julia --project=. -t 2 scripts/m5_bifrost/codex_m5b_review_checks.jl
# No production methods/files are replaced. A renamed copy of the pressure block
# is instrumented in this Julia process; its iterates are checked against production.
using HSEMHS, MPI, Printf, Test
const H = HSEMHS
H.ensure_mpi()
@assert MPI.Comm_size(MPI.COMM_WORLD) == 1 "Use one rank for these small examples."

# Re-run Claude's targeted regression testset, without the unrelated long suite.
testpath = normpath(joinpath(@__DIR__, "..", "..", "test", "test_stag_gradient.jl"))
testsrc = read(testpath, String)
marker = findfirst("@testset \"M5: constrained pressure projection and block termination", testsrc)
@assert marker !== nothing
Base.include_string(Main, testsrc[marker.start:end], testpath)

@eval H const CODEX_M5B_TRACE = NamedTuple[]
srcpath = normpath(joinpath(@__DIR__, "..", "..", "src", "MHSAlternate.jl"))
src = read(srcpath, String)
start = findfirst("function u_gauss_newton!", src).start
stop = findfirst("\n\"\"\"\n    alternating_solve!", src).start - 1
src = replace(src[start:stop], "function u_gauss_newton!" => "function codex_m5b_audited_u!")
src = replace(src, "    f = fg!(g, pb, w, x)[1]" => "    basemodel = SMHSWork(par); modeltmp = SMHSWork(par); modelHz = similar_state(x); modelz = similar_state(x)\n    f = fg!(g, pb, w, x)[1]")
src = replace(src, "        tries += 1" => "        objective!(pb, basemodel, x)\n        tries += 1")
src = replace(src, "            Hv!(q, hd)\n            pred =" => "            copyto!(parent(modelz[1]), parent(hd)); gn_hvp!(modelHz, pb, basemodel, modeltmp, modelz)\n            refpred = -(_dot1(par, g[1], hd) + 0.5 * _dot1(par, hd, modelHz[1]))\n            drift = maximum(abs(w.u[i,j,k] - basemodel.u[i,j,k]) for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx)\n            actualnorm = sqrt(sum(D1[i,j,k] * hd[i,j,k]^2 for k in 1:b.nz, j in 1:b.ny, i in 1:b.nx))\n            Hv!(q, hd)\n            pred =")
src = replace(src, "            ρ > 1e-4 && ft < f && margin >= -1e-12 && break" => "            push!(CODEX_M5B_TRACE, (tries=tries, tarc=tarc, drift=drift, pred=pred, refpred=refpred, ratio=actualnorm/Δ, rho=ρ, accepted=ρ > 1e-4 && ft < f && margin >= -1e-12))\n            ρ > 1e-4 && ft < f && margin >= -1e-12 && break")
@assert occursin("push!(CODEX_M5B_TRACE", src) && occursin("refpred =", src)
Base.include_string(H, src, "codex_m5b_instrumented_pressure_block")

function review_pressure_block()
    par = H.init_parallel(4,4,9)
    pb, x0, us, uc, grid = H.sc1_problem(par,4,4,9; lambda_Q=1e-3)
    @testset "M5b: reproduce model drift and trust-region violation" begin
        for amp in (0.8,4.0)
            x = deepcopy(x0)
            H.owned_map!((i,j,k)->x[1][i,j,k] + amp*sin(1.37i+2.21j+0.7k),x[1],par)
            xoriginal = deepcopy(x)
            w = H.SMHSWork(par); H.objective!(pb,w,x)
            ulo=H.newfield(par); H.owned_map!((i,j,k)->w.u[i,j,k]-3.0,ulo,par)
            empty!(H.CODEX_M5B_TRACE)
            r=H.codex_m5b_audited_u!(pb,w,x,ulo;steps=5,cg_iter=100,cg_rtol=1e-4,du_max=0.5,coord=:lnP,record=true)
            trace=copy(H.CODEX_M5B_TRACE)
            woriginal = H.SMHSWork(par); H.objective!(pb,woriginal,xoriginal)
            ro=H.u_gauss_newton!(pb,woriginal,xoriginal,ulo;steps=5,cg_iter=100,cg_rtol=1e-4,du_max=0.5,coord=:lnP,record=true)
            # Instrumentation must not change the actual solver trajectory.
            @test r.f ≈ ro.f rtol=1e-13
            @test maximum(abs.(parent(x[1]) - parent(xoriginal[1]))) <= 1e-12
            stale=filter(q->q.drift>1e-10,trace)
            @test !isempty(stale)
            @test any(q->q.pred<0 && q.refpred>0,stale)
            excessive=filter(q->q.accepted && q.ratio>1.00001,trace)
            @test !isempty(excessive)
            @printf("amp=%.1f: %d arc trials, %d use a changed model state\n",amp,length(trace),length(stale))
            println("  changed-model example: ",stale[1])
            println("  accepted outside trust region: ",excessive[1])
            flush(stdout)
        end
    end
    # This is the current magnetic-block acceptance predicate, independent of
    # whether its work ended in an actual line-search failure.
    ra = (status=H.stalled,iterations=1)
    a_ok = ra.status == H.converged || (ra.iterations > 0 && ra.status != H.eos_failure)
    @test a_ok
    println("Magnetic block predicate accepts stalled after one iteration: ",a_ok)
end

review_pressure_block()
