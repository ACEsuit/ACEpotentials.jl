# Pure Julia benchmark for ETACE export performance
# Run with: julia --project=. benchmark/julia_benchmark.jl

using ACEpotentials
using ACEpotentials.Models
using ACEpotentials.ETModels
using StaticArrays
using Random
using Lux
using LuxCore
using BenchmarkTools

const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels

println("="^70)
println("ETACE Export Performance Benchmark")
println("="^70)

# Create model matching the benchmark configuration
elements = (:Ti, :Al)
order = 3
totaldegree = 10
rcut = 5.5
maxl = 2

println("\nModel configuration:")
println("  elements: $elements")
println("  order: $order")
println("  totaldegree: $totaldegree")
println("  rcut: $rcut")
println("  maxl: $maxl")

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
n_basis = length(ps.WB[:, 1])
println("  basis size: $n_basis")

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
mkpath("/tmp/benchmark_export")
export_ace_model(et_calc, "/tmp/benchmark_export/spline.jl"; for_library=true, radial_basis=:spline)
export_ace_model(et_calc, "/tmp/benchmark_export/poly.jl"; for_library=true, radial_basis=:polynomial)

# Load the exported modules
println("\nLoading exported modules...")
spline_mod = Module(:SplineExport)
Base.include(spline_mod, "/tmp/benchmark_export/spline.jl")

poly_mod = Module(:PolyExport)
Base.include(poly_mod, "/tmp/benchmark_export/poly.jl")

# Create realistic test data (matching ~112 neighbors at rcut=5.5)
function generate_bulk_neighbors(n_atoms::Int, rcut::Float64; seed=42)
    rng = MersenneTwister(seed)
    Rs = Vector{SVector{3, Float64}}()
    Zs = Vector{Int}()

    # Generate neighbors in a shell
    for _ in 1:n_atoms
        # Random direction
        theta = 2π * rand(rng)
        phi = acos(2*rand(rng) - 1)
        # Random distance between 2.0 and rcut
        r = 2.0 + (rcut - 2.0) * rand(rng)
        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)
        push!(Rs, SVector(x, y, z))
        push!(Zs, rand(rng, [22, 13]))  # Ti or Al
    end
    return Rs, Zs
end

# Benchmark configuration
n_neighbors_list = [50, 100, 150]  # Different neighbor counts
n_sites = 100  # Number of sites to average over

println("\n" * "="^70)
println("Benchmark Results")
println("="^70)

for n_neighbors in n_neighbors_list
    println("\n--- $n_neighbors neighbors ---")

    # Generate test configurations
    configs = [generate_bulk_neighbors(n_neighbors, rcut; seed=i) for i in 1:n_sites]

    # Warm-up
    for (Rs, Zs) in configs[1:5]
        spline_mod.site_energy_forces(Rs, Zs, 22)
        poly_mod.site_energy_forces(Rs, Zs, 22)
    end

    # Benchmark spline version
    t_spline = @benchmark begin
        for (Rs, Zs) in $configs
            $spline_mod.site_energy_forces(Rs, Zs, 22)
        end
    end samples=5

    # Benchmark polynomial version
    t_poly = @benchmark begin
        for (Rs, Zs) in $configs
            $poly_mod.site_energy_forces(Rs, Zs, 22)
        end
    end samples=5

    # Report
    spline_per_site = median(t_spline).time / 1e6 / n_sites  # ms per site
    poly_per_site = median(t_poly).time / 1e6 / n_sites

    println("  Spline:     $(round(spline_per_site, digits=3)) ms/site")
    println("  Polynomial: $(round(poly_per_site, digits=3)) ms/site")
    println("  Speedup:    $(round(poly_per_site / spline_per_site, digits=2))x (spline faster)")
end

# Single-threaded throughput estimate
println("\n" * "="^70)
println("Throughput Estimate (single thread)")
println("="^70)

n_neighbors = 112  # Typical bulk coordination
Rs, Zs = generate_bulk_neighbors(n_neighbors, rcut; seed=999)

t_single = @benchmark $spline_mod.site_energy_forces($Rs, $Zs, 22) samples=100

time_per_site = median(t_single).time / 1e9  # seconds
atoms_per_second = 1.0 / time_per_site

println("  Time per site:        $(round(time_per_site * 1e3, digits=3)) ms")
println("  Sites per second:     $(round(atoms_per_second, digits=1))")
println("  For 2000 atoms:       $(round(2000/atoms_per_second, digits=2)) s per timestep")
println("  For 100 timesteps:    $(round(100 * 2000/atoms_per_second, digits=1)) s total")

# Verify numerical correctness
println("\n" * "="^70)
println("Numerical Verification")
println("="^70)

Rs, Zs = generate_bulk_neighbors(50, rcut; seed=123)
E_spline, F_spline = spline_mod.site_energy_forces(Rs, Zs, 22)
E_poly, F_poly = poly_mod.site_energy_forces(Rs, Zs, 22)

energy_err = abs(E_spline - E_poly)
force_err = maximum(norm.(F_spline .- F_poly))

println("  Energy error: $(round(energy_err, sigdigits=3)) eV")
println("  Max force error: $(round(force_err, sigdigits=3)) eV/Å")
println("  Status: ", energy_err < 1e-5 && force_err < 1e-4 ? "✓ PASS" : "✗ FAIL")

println("\n" * "="^70)
println("Benchmark Complete")
println("="^70)
