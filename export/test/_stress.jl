include(joinpath(@__DIR__, "check_export.jl"))
using StaticArrays, LinearAlgebra
fx = load_cantor_fixture()
ex = Base.invokelatest(load_exported, joinpath(@__DIR__, "build", "cantor_ws_poly.jl"))
sets = Base.invokelatest(site_sets, fx.held[1], fx.rcut)
function stress(ex, sets, nrep)
    ws = Base.invokelatest(ex.new_workspace)
    acc = 0.0
    for r in 1:nrep
        for (Rs, Zs, Z0, _) in sets
            F = Vector{SVector{3,Float64}}(undef, length(Rs))
            E, V = Base.invokelatest(ex.site_energy_forces_virial!, ws, Rs, Zs, Z0, F)
            acc += E + F[1][1] + V[1,1]
        end
        if r % 200 == 0
            GC.gc()
            println("rep $r  acc=$acc"); flush(stdout)
        end
    end
    return acc
end
println("nsites = ", length(sets), ", max nneigh = ", maximum(length(s[1]) for s in sets))
@time stress(ex, sets, 2000)
println("STRESS OK")
