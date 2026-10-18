# Read-only saved-state diagnostics for CODEX_REVIEW_M5c_2026-10-18.md.
# Run from HSEMHS with: julia --project=. -t 4 scripts/m5_bifrost/codex_m5c_result_review.jl
# Requires the existing out/m5c checkpoints. Reconstructing the original
# initializer can take several minutes. No solver comparison is rerun.
include(joinpath(@__DIR__, "bif_setup.jl"))
using Test
function saved_review()
    pb,xinit,par,b,NZ=bif_problem(12,2;lambda_F=1e-4,lambda_Q=1e-3)
    @assert MPI.Comm_size(par.comm)==1
    root="/Users/harshm/Documents/WorkRepo/MHS/HSEMHS/out/m5c"
    states=Dict{String,typeof(xinit)}("initializer"=>deepcopy(xinit))
    uinit=H.newfield(par)
    for (tag,file) in (("checkpoint","ck.h5"),("alt","end_alt.h5"),("joint","end_joint.h5"))
        x=deepcopy(xinit)
        h5open(joinpath(root,file)) do f
            for q in 1:5;copyto!(parent(x[q]),read(f["x$q"]));end
            tag=="checkpoint" && copyto!(parent(uinit),read(f["u_init"]))
        end
        H.exchange_halos!(par,collect(x));states[tag]=x
    end
    ulo=H.newfield(par);ma=H.anchor_mean(pb,uinit)
    H.owned_map!((i,j,k)->uinit[i,j,k]-ma+pb.lnPref-3.0,ulo,par)
    w=H.SMHSWork(par)
    H.objective!(pb,w,states["checkpoint"])
    dPu=H.dP_unobs!(H.newfield(par),pb,w)
    ident=[dPu[i,j,k]<=0.1*w.P[i,j,k] for i in 1:b.nx,j in 1:b.ny,k in 1:NZ]
    metrics=Dict()
    for tag in ("checkpoint","alt","joint")
        H.objective!(pb,w,states[tag]);metrics[tag]=H.reference_metric(pb,w,states[tag])
    end
    for tag in ("initializer","checkpoint","alt","joint")
        x=states[tag];f,t,_=H.objective!(pb,w,x)
        devs=Tuple{Float64,Int,Int,Int,Int,Float64,Float64,Float64}[]
        for c in 1:3,k in axes(w.B[c],3),j in 1:b.ny,i in 1:b.nx
            sig=1/sqrt(pb.invsf2[c][i,j,k]);bv=w.B[c][i,j,k];bo=pb.Bobsf[c][i,j,k]
            push!(devs,(abs((bv-bo)/sig),c,i,j,k,bv,bo,sig))
        end
        sort!(devs,rev=true)
        println("STATE ",tag," f=",f," rawF=",t[1]/pb.lamF," data=",t[3]," Q=",t[2]," top outliers=",devs[1:3]);flush(stdout)
        tag=="initializer" && continue
        for met in ("checkpoint","alt","joint")
            a=H.full_audit(pb,w,x,metrics[met],ulo;ident=ident,nworst=1)
            @printf("AUDIT state=%s metric=%s m=%.6e u=%.6e A=%.6e ident=%.6e rest=%.6e\n",tag,met,a.m,a.m_u,a.m_A,a.u_max_ident,a.u_max_rest)
        end
        # Along the original audit correction at the worst complement node,
        # compare the old diagonal against current local curvature and gradient.
        a=H.full_audit(pb,w,x,metrics["checkpoint"],ulo;ident=ident,nworst=1)
        node=a.worst[1];i,j,k=node.i,node.j,node.k
        @printf("WORST state=%s i=%d j=%d k=%d P/P_ref=%.5e g=%.5e D_old=%.5e D_current=%.5e ratio=%.5e\n",tag,i,j,k,node.P_over_Pref,node.g,node.D,metrics[tag][1][i,j,k],metrics[tag][1][i,j,k]/node.D)
        flush(stdout)
    end
    # Necessary compatibility checks for a divergence-free field within t sigma.
    # Each cell gives |div Bobs| <= t sum_f(sigma_f / h_f).
    H.exchange_halos!(par,collect(pb.Bobsf));H.exchange_halos!(par,collect(pb.invsf2))
    worst=(0.0,0,0,0)
    for k in 1:NZ,j in 1:b.ny,i in 1:b.nx
        sig(c,i,j,k)=1/sqrt(pb.invsf2[c][i,j,k])
        den=(sig(1,i,j,k)+sig(1,i-1,j,k))/pb.s.dxc[i]+(sig(2,i,j,k)+sig(2,i,j-1,k))/pb.s.dyc[j]+(sig(3,i,j,k-1)+sig(3,i,j,k))/pb.s.hz[k]
        need=abs(H.div_face(pb.Bobsf,pb.s,i,j,k))/den
        need>worst[1] && (worst=(need,i,j,k))
    end
    println("NECESSARY local box threshold sigma >= ",worst)
    means=[sum(pb.Bobsf[3][i,j,k] for i in 1:b.nx,j in 1:b.ny)/(b.nx*b.ny) for k in 0:NZ]
    sigmeans=[sum(1/sqrt(pb.invsf2[3][i,j,k]) for i in 1:b.nx,j in 1:b.ny)/(b.nx*b.ny) for k in 0:NZ]
    need=maximum(abs(means[k]-means[l])/(sigmeans[k]+sigmeans[l]) for k in eachindex(means),l in eachindex(means))
    println("NECESSARY plane-mean threshold sigma >= ",need,"; Bz mean range=",extrema(means))
end
saved_review()
