"""
Performance comparison: Current hybrid approach vs pure Zygote for force computation

Current approach: Custom pullback using EquivariantTensors.pullback! + ForwardDiff for radial/angular bases
Zygote approach: Pure reverse-mode AD through entire model

Run with: julia --project=. benchmark_ad_approaches.jl
"""

using ACEpotentials
using BenchmarkTools
using Zygote
using StaticArrays
using Printf
using ExtXYZ
using LazyArtifacts

println("=" ^ 80)
println("AD Approach Performance Comparison")
println("=" ^ 80)
println()

# Setup: Create a small test system
println("Setting up test system...")
model = ace1_model(;
    elements = [:Si],
    order = 3,
    totaldegree = 12,
    rcut = 5.5,
    Eref = [:Si => -158.54496821]
)

println("Model: order=3, totaldegree=12, rcut=5.5Å")
println()

# Load data and fit model
println("Loading dataset and fitting model...")
data = ExtXYZ.load(artifact"Si_tiny_dataset" * "/Si_tiny.xyz")

data_keys = [:energy_key => "dft_energy",
             :force_key => "dft_force",
             :virial_key => "dft_virial"]

train_data = [ACEpotentials.AtomsData(at; data_keys...) for at in data]

solver_qr = ACEfit.LSQR(; damp=1e-3, atol=1e-6)
acefit!(train_data, model; data_keys..., solver=solver_qr)

println("Model fitted successfully")
println()

# Extract model components for direct evaluation
ace_model = model.model
ps = model.ps
st = model.st

# Get a single site configuration for benchmarking
at_test = train_data[1].config
nlist = ACEpotentials.Calculators.nlist(at_test, model)
i_site = 1
Rs, Zs, Z0 = ACEpotentials.Models.get_rpi(nlist, at_test, i_site)

println("Benchmark configuration:")
println("  Site atom: ", Z0)
println("  Neighbors: ", length(Rs))
println()

# ============================================================================
# Approach 1: Current hybrid (custom pullback + ForwardDiff)
# ============================================================================

function forces_current(model, Rs, Zs, Z0, ps, st)
    """Current approach: Custom pullback with ForwardDiff for radial/angular"""
    E, ∇E = ACEpotentials.Models.energy_and_grad(model, Rs, Zs, Z0, ps, st)
    return ∇E
end

# ============================================================================
# Approach 2: Pure Zygote
# ============================================================================

function energy_only(model, Rs, Zs, Z0, ps, st)
    """Energy evaluation only (for Zygote to differentiate)"""
    # Note: Zygote doesn't like mutation, so we use the simpler evaluate path
    return ACEpotentials.Models.energy(model, Rs, Zs, Z0, ps, st)
end

function forces_zygote(model, Rs, Zs, Z0, ps, st)
    """Pure Zygote approach: Reverse-mode AD through entire model"""
    # Zygote differentiates w.r.t. first argument (Rs)
    grads = Zygote.gradient(Rs -> energy_only(model, Rs, Zs, Z0, ps, st), Rs)
    return grads[1]
end

# ============================================================================
# Verification: Check both approaches give same results
# ============================================================================

println("Verification: Checking numerical equivalence...")
println()

f_current = forces_current(ace_model, Rs, Zs, Z0, ps, st)
f_zygote = forces_zygote(ace_model, Rs, Zs, Z0, ps, st)

max_diff = maximum(abs.(f_current .- f_zygote))
rel_diff = max_diff / maximum(abs, f_current)

println(@sprintf("  Max absolute difference: %.3e", max_diff))
println(@sprintf("  Max relative difference: %.3e", rel_diff))

if rel_diff < 1e-6
    println("  ✓ Results match (within numerical precision)")
else
    println("  ⚠ WARNING: Significant difference detected!")
    println("  This may indicate a problem with one of the implementations")
end
println()

# ============================================================================
# Benchmarking
# ============================================================================

println("=" ^ 80)
println("Performance Benchmarks")
println("=" ^ 80)
println()

println("Approach 1: Current (custom pullback + ForwardDiff)")
println("-" ^ 80)
b1 = @benchmark forces_current($ace_model, $Rs, $Zs, $Z0, $ps, $st) samples=1000 evals=10
display(b1)
println()
println()

println("Approach 2: Pure Zygote")
println("-" ^ 80)
b2 = @benchmark forces_zygote($ace_model, $Rs, $Zs, $Z0, $ps, $st) samples=1000 evals=10
display(b2)
println()
println()

# ============================================================================
# Summary
# ============================================================================

println("=" ^ 80)
println("Summary")
println("=" ^ 80)
println()

median_current = median(b1.times) / 1e6  # Convert to ms
median_zygote = median(b2.times) / 1e6
speedup = median_zygote / median_current

alloc_current = b1.allocs
alloc_zygote = b2.allocs

println(@sprintf("%-30s %10.3f ms  (%d allocs)", "Current (hybrid):", median_current, alloc_current))
println(@sprintf("%-30s %10.3f ms  (%d allocs)", "Pure Zygote:", median_zygote, alloc_zygote))
println()
println(@sprintf("Speedup (Current vs Zygote): %.2fx", speedup))
println(@sprintf("Allocation ratio: %.2fx", alloc_zygote / max(alloc_current, 1)))
println()

if speedup > 1.5
    println("✓ Current approach is significantly faster (>1.5x)")
    println("  Recommendation: Keep current hybrid approach for production")
elseif speedup > 1.1
    println("≈ Current approach is moderately faster (1.1-1.5x)")
    println("  Recommendation: Current approach preferable, but Zygote viable for prototyping")
else
    println("✓ Approaches have similar performance (<1.1x difference)")
    println("  Recommendation: Consider migrating to Zygote for simplicity")
end
println()

println("=" ^ 80)
println("Additional Notes")
println("=" ^ 80)
println()
println("• Current approach uses custom pullback with manual chain rule assembly")
println("• Zygote provides automatic differentiation with ChainRules integration")
println("• Performance may vary with model size and system configuration")
println("• Memory usage typically favors current approach due to @no_escape/@withalloc")
println()
