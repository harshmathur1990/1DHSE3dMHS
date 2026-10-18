# Independent checks for CODEX_REVIEW_M5_SECOND_FOLLOWUP.md.
# Run from HSEMHS: julia --project=. -t 2 scripts/m5_bifrost/codex_m5c_review_checks.jl
# Optional argument: floor (runs the six-cycle synthetic C1b diagnostic instead
# of the small coordinate probes; this takes several minutes).
# The instrumented routine has a new name. Production source/methods are unchanged.
using HSEMHS, MPI, Test, Printf
const H=HSEMHS
H.ensure_mpi()
rank0(par)=H.rank0(par)
ts=read("test/test_stag_gradient.jl",String)
k=findfirst("@testset \"M5: constrained pressure projection",ts).start
Base.include_string(Main,ts[k:end],"M5_targeted_regression_tests")
@eval H const CODEX_CAND_TRACE=NamedTuple[]
src=read("src/MHSAlternate.jl",String)
a=findfirst("function u_gauss_newton!",src).start
b=findfirst("\n\"\"\"Scaled optimality",src).start-1
src=replace(src[a:b],"function u_gauss_newton!"=>"function codex_m5c_u!")
src=replace(src,"        for _ in 1:cg_iter"=>"        trace_ncg=0; trace_boundary=false\n        for _ in 1:cg_iter\n            trace_ncg += 1")
src=replace(src,"            if α >= τt"=>"            if α >= τt\n                trace_boundary=true\n                #")
src=replace(src,"        aerr = candidate!(δ, y, 1.0, Δ)"=>"        trace_h = [sc[i,j,k]*y[i,j,k] for i in 1:b.nx,j in 1:b.ny,k in 1:b.nz]\n        aerr = candidate!(δ, y, 1.0, Δ)")
src=replace(src,"        if predc > 0 && pred < 0.1 * predc"=>"        push!(CODEX_CAND_TRACE,(h=trace_h,pred=pred,predc=predc,min_positive=minimum(1 .+ trace_h),clipped=count(v->v < -0.999999,trace_h),ncg=trace_ncg,boundary=trace_boundary,cg_rel=cg_rel))\n        if predc > 0 && pred < 0.1 * predc")
Base.include_string(H,src,"codex_candidate_trace")
function probes()
  par=H.init_parallel(4,4,9)
  pb,x0,us,uc,grid=H.sc1_problem(par,4,4,9;lambda_Q=1e-3)
  for amp in (0.8,4.0)
    traces=[]
    for coord in (:lnP,:P)
      x=deepcopy(x0); H.owned_map!((i,j,k)->x[1][i,j,k]+amp*sin(1.37i+2.21j+0.7k),x[1],par)
      w=H.SMHSWork(par); H.objective!(pb,w,x)
      ulo=H.newfield(par); H.owned_map!((i,j,k)->w.u[i,j,k]-3.0,ulo,par)
      empty!(H.CODEX_CAND_TRACE)
      r=H.codex_m5c_u!(pb,w,x,ulo;steps=1,cg_iter=100,cg_rtol=1e-4,coord=coord,record=true)
      tr=H.CODEX_CAND_TRACE[1]; push!(traces,tr)
      @printf("amp=%.1f coord=%s first PCG candidate: min(1+h)=%.4e positivity clips=%d pred/predc=%.4e; accepted=%d, candidates=%d, fallbacks=%d; ncg=%d boundary=%s rel=%.3e\n",amp,coord,tr.min_positive,tr.clipped,tr.pred/tr.predc,r.accepted,r.counts.cand,r.counts.cauchy,tr.ncg,tr.boundary,tr.cg_rel)
    end
    @printf("amp=%.1f relative difference of the initial linearized h directions: %.3e\n",amp,maximum(abs.(traces[1].h-traces[2].h))/maximum(abs.(traces[1].h)))
  end
end
function floor_probe()
  par=H.init_parallel(8,8,65)
  pb,x,us,uc,grid=H.sc1_problem(par,8,8,65;Bobs=(20e-4,0.0,50e-4))
  w=H.SMHSWork(par); g=H.similar_state(x)
  ga0=H.atilde_gradnorm(pb,w,x,g)
  st,why,n,hist=H.alternating_solve!(pb,w,x;max_outer=6,a_iter=300,u_steps=5,verbose=true)
  f,t,_=H.objective!(pb,w,x); rms,p99=H.layer_residual_stats(pb,w)
  ga=H.atilde_gradnorm(pb,w,x,g)
  db=maximum(maximum(abs(w.B[q][i,j,k]-pb.Bobsf[q][i,j,k])*sqrt(pb.invsf2[q][i,j,k]) for i in 1:8,j in 1:8,k in axes(w.B[q],3)) for q in 1:3)
  du=maximum(abs(w.u[i,j,k]-us[i,j,k]) for i in 1:8,j in 1:8,k in 1:65)
  @printf("C1b: status=%s, cycles=%d, phi=%.9e, maxdu=%.3e, maxRMS=%.3e, maxp99=%.3e, maxdB/sigma=%.3e, ga0=%.3e, ga=%.3e, arel=%.3e\n",string(st),n,f,du,maximum(rms),maximum(p99),db,ga0,ga,ga/ga0)
end

if "floor" in ARGS
  floor_probe()
else
  probes()
end
