# M5f review reproducer: fixed central-difference probes can leave the log-barrier domain.
# From HSEMHS/: julia --project=. -t 2 scripts/m5_bifrost/codex_m5f_review_checks.jl
# Reports the current preconditioner versus an exact base-state GN product, then runs the focused tests.
using HSEMHS,MPI,Statistics,Printf,Test
const H=HSEMHS
H.ensure_mpi()
par=H.init_parallel(4,4,9);b=par.block
pb,x,us,uc,grid=H.sc1_problem(par,4,4,9;Bobs=(0.,0.,.005),lambda_F=1e-4,lambda_Q=1e-3)
sig=1/sqrt(pb.invsf2[3][1,1,0])
H.setscalar!(par,x[5],.005+(3.0 - 1e-6)*sig)
hard=(H.newfield(par;fill=3.),H.newfield(par;fill=3.),H.newfield0(par;fill=3.))
H.set_face_hinge!(pb,hard;lambda=.1,kind=:barrier)
w=H.SMHSWork(par);f,t,_=H.objective!(pb,w,x)
v=H.similar_state(x);H.setscalar!(par,v[5],1.)
hv=H.similar_state(x);H.gn_hvp!(hv,pb,w,H.SMHSWork(par),v)
exact=H.getscalar(par,hv[5]);eb=1e-6*max(abs(H.getscalar(par,x[5])),1e-4)
xp=deepcopy(x);xm=deepcopy(x)
H.lincomb!(xp,1.,x,eb,v,par);H.lincomb!(xm,1.,x,-eb,v,par)
gp=H.similar_state(x);gm=H.similar_state(x)
fp=H.fg!(gp,pb,H.SMHSWork(par),xp)[1];fm=H.fg!(gm,pb,H.SMHSWork(par),xm)[1]
fd=(H.getscalar(par,gp[5])-H.getscalar(par,gm[5]))/(2eb)
H.objective!(pb,w,x);D=H.gn_diagonal!(H.similar_state(x),pb,w,x)
@printf("base finite %s, min slack %.3g sigma; bbar probe %.3g sigma; fplus %g fminus %g\n",isfinite(f),1e-6,eb/sig,fp,fm)
@printf("bbar curvature: exact GN %.9e, raw FD %.9e, used |FD| %.9e; ratio used/exact %.9e\n",exact,fd,H.getscalar(par,D[5]),H.getscalar(par,D[5])/exact)
@test isfinite(f) && !isfinite(fp) && isfinite(fm)
@test exact>0 && fd<0
# Existing focused tests exercise the default-disabled behavior and anchor-tangent path.
src=read("test/test_stag_gradient.jl",String);i=findfirst("@testset \"M5d:",src)
include_string(Main,src[first(i):end],"test/test_stag_gradient.jl")
