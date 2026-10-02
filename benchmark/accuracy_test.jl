# Accuracy verification: Compare exported models against native ETACE
# Run with: julia --project=. benchmark/accuracy_test.jl

using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using StaticArrays
using Random
using Lux
using LuxCore
using LinearAlgebra
using Statistics

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

println("="^70)
println("ETACE Export Accuracy Verification")
println("="^70)

# Create model
elements = (:Ti, :Al)
order = 3
totaldegree = 10
rcut = 5.5
maxl = 2

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)
rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(;
    elements = elements,
    order = order,
    Ytype = :solid,
    level = M.TotalDegree(),
    max_level = totaldegree,
    maxl = maxl,
    pair_maxn = totaldegree,
    rin0cuts = rin0cuts,
    init_WB = :glorot_normal,
    init_Wpair = :glorot_normal
)

ps, st = Lux.setup(rng, ace_model)

# Convert to ETACE
et_model = ETM.convert2et(ace_model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

n_species = length(elements)
for iz in 1:n_species
    for jz in 1:n_species
        et_ps.rembed.post.W[:, :, (iz-1)*n_species + jz] .= ps.rbasis.Wnlq[:, :, iz, jz]
    end
end
for iz in 1:n_species
    et_ps.readout.W[1, :, iz] .= ps.WB[:, iz]
end

et_calc = ETM.ETACEPotential(et_model, et_ps, et_st, rcut)

# Include export code
include(joinpath(@__DIR__, "..", "export/src/export_ace_model.jl"))

# Export both versions
println("\nExporting models...")
mkpath("/tmp/accuracy_test")
export_ace_model(et_calc, "/tmp/accuracy_test/spline.jl"; for_library=true, radial_basis=:spline, n_spline_samples=2000)
export_ace_model(et_calc, "/tmp/accuracy_test/poly.jl"; for_library=true, radial_basis=:polynomial)

# Load the exported modules
spline_mod = Module(:SplineExport)
Base.include(spline_mod, "/tmp/accuracy_test/spline.jl")

poly_mod = Module(:PolyExport)
Base.include(poly_mod, "/tmp/accuracy_test/poly.jl")

# Generate test configurations
function generate_test_config(n_neighbors::Int, rcut::Float64; seed=42)
    rng = MersenneTwister(seed)
    Rs = Vector{SVector{3, Float64}}()
    Zs = Vector{Int}()

    for _ in 1:n_neighbors
        theta = 2π * rand(rng)
        phi = acos(2*rand(rng) - 1)
        r = 2.0 + (rcut - 2.0) * rand(rng)
        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)
        push!(Rs, SVector(x, y, z))
        push!(Zs, rand(rng, [22, 13]))
    end
    return Rs, Zs
end

println("\n" * "="^70)
println("Accuracy Results: Spline vs Polynomial")
println("="^70)

energy_errors = Float64[]
force_errors = Float64[]

for seed in 1:100
    Rs, Zs = generate_test_config(100, rcut; seed=seed)

    E_spline, F_spline = spline_mod.site_energy_forces(Rs, Zs, 22)
    E_poly, F_poly = poly_mod.site_energy_forces(Rs, Zs, 22)

    push!(energy_errors, abs(E_spline - E_poly))
    push!(force_errors, maximum(norm.(F_spline .- F_poly)))
end

println("\nEnergy error statistics (eV):")
println("  Mean:   $(round(mean(energy_errors), sigdigits=3))")
println("  Max:    $(round(maximum(energy_errors), sigdigits=3))")
println("  Std:    $(round(std(energy_errors), sigdigits=3))")

println("\nForce error statistics (eV/Å):")
println("  Mean:   $(round(mean(force_errors), sigdigits=3))")
println("  Max:    $(round(maximum(force_errors), sigdigits=3))")
println("  Std:    $(round(std(force_errors), sigdigits=3))")

# Check if errors are within acceptable range for MD
# Typical thresholds: energy ~ 1 meV/atom, force ~ 10 meV/Å
energy_ok = maximum(energy_errors) < 0.001  # 1 meV
force_ok = maximum(force_errors) < 0.01     # 10 meV/Å

println("\nAcceptability check:")
println("  Energy < 1 meV:    ", energy_ok ? "✓ PASS" : "✗ FAIL ($(round(maximum(energy_errors)*1000, digits=2)) meV)")
println("  Force < 10 meV/Å:  ", force_ok ? "✓ PASS" : "✗ FAIL ($(round(maximum(force_errors)*1000, digits=2)) meV/Å)")

# Overall status
if energy_ok && force_ok
    println("\n✓ Spline interpolation accuracy is ACCEPTABLE for MD simulations")
else
    println("\n⚠ Spline interpolation may need more sample points for high-accuracy work")
    println("  Current: n_spline_samples=2000")
    println("  Consider: n_spline_samples=5000 for higher accuracy")
end

println("\n" * "="^70)
