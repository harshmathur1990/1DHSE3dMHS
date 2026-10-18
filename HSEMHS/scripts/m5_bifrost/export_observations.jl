# One shared observation artifact for the Julia solver and the LP checks (Codex M5d review, step 2): the exact
# arrays bif_setup.jl uses. Face observations I·y and face σ (geometric mean), centre observations y and centre
# σ, and the grid metrics. Index conventions (1-based here, saved as stored):
#   x-faces (nx, ny, nz): face i lies between centres i and i+1 (periodic); y-faces likewise;
#   z-faces (nx, ny, nz+1): face k (0-based in the file: index 1 = top boundary) lies above centre k;
#   centre values (nx, ny, nz), k = 1 at the top.
# usage (from HSEMHS/): julia --project=. -t 4 scripts/m5_bifrost/export_observations.jl out/m5d/observations.h5
include(joinpath(@__DIR__, "bif_setup.jl"))
out = ARGS[1]
pb, x, par, b, NZ = bif_problem(12, 2; lambda_F = 1e-4, lambda_Q = 1e-3)
h5open(out, "w") do f
    for c in 1:3
        v = pb.Bobsf[c]; iv = pb.invsf2[c]; ks = collect(axes(v, 3))
        f["face_obs$c"] = [v[i, j, k] for i in 1:b.nx, j in 1:b.ny, k in ks]
        f["face_sigma$c"] = [1 / sqrt(iv[i, j, k]) for i in 1:b.nx, j in 1:b.ny, k in ks]
        f["centre_obs$c"] = [pb.Bobs[c][i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
        f["centre_sigma$c"] = [pb.sig[c][i, j, k] for i in 1:b.nx, j in 1:b.ny, k in 1:NZ]
    end
    f["dxc"] = pb.s.dxc[1:b.nx]; f["dyc"] = pb.s.dyc[1:b.ny]; f["hz"] = pb.s.hz[1:NZ]
    attributes(f)["source"] = "bif_setup.jl bif_problem(12, 2): 12×12 block means (Float32 input), every 2nd level"
end
println("exported $(out)")
