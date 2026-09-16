include(joinpath(@__DIR__, "check_export.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))
using StaticArrays, LinearAlgebra
fx = load_cantor_fixture()
f = joinpath(@__DIR__, "build", "cantor_ws_poly.jl")
ex = Base.invokelatest(load_exported, f)
sets = Base.invokelatest(site_sets, fx.held[1], fx.rcut)
ref = [Base.invokelatest(ex.site_energy_forces_virial, s[1], s[2], s[3]) for s in sets]
n = 4
wss = [Base.invokelatest(ex.new_workspace) for _ in 1:n]
out = Vector{Any}(undef, length(sets))
tasks = map(1:n) do c
    Threads.@spawn begin
        w = wss[c]
        for i in c:n:length(sets)
            Rs, Zs, Z0, _ = sets[i]
            F = Vector{SVector{3,Float64}}(undef, length(Rs))
            E, V = Base.invokelatest(ex.site_energy_forces_virial!, w, Rs, Zs, Z0, F)
            out[i] = (E, F, V)
        end
    end
end
foreach(wait, tasks)
let dE = 0.0, dF = 0.0, dV = 0.0
for i in eachindex(sets)
    dE = max(dE, abs(out[i][1] - ref[i][1]))
    dF = max(dF, maximum(norm.(out[i][2] .- ref[i][2])))
    dV = max(dV, maximum(abs.(out[i][3] .- ref[i][3])))
end
nb=0; for i in eachindex(sets); ok = (out[i][1] === ref[i][1]) && (out[i][2] == ref[i][2]) && (out[i][3] == ref[i][3]); ok || (nb+=1); end
println("dE=", dE, " dF=", dF, " dV=", dV, " nbad=", nb, "  BLAS threads=", BLAS.get_num_threads())
end
