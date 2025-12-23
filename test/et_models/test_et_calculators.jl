# Tests for ETACEPotential calculator interface
#
# These tests verify:
# 1. Energy consistency between ETACE model and ETACEPotential
# 2. Force consistency against original ACE model
# 3. Virial consistency against original ACE model
# 4. AtomsCalculators interface compliance

using Test, ACEbase, BenchmarkTools
using Polynomials4ML.Testing: print_tf, println_slim

using ACEpotentials
M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

import EquivariantTensors as ET
import AtomsCalculators

using AtomsBase, AtomsBuilder, Unitful
using Random, LuxCore, StaticArrays, LinearAlgebra

rng = Random.MersenneTwister(1234)
Random.seed!(1234)

##
# Build an ETACE model for testing

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 10
order = 3
maxl = 6

# Use same cutoff for all elements
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)

model = M.ace_model(; elements = elements, order = order,
                     Ytype = :solid, level = level, max_level = max_level,
                     maxl = maxl, pair_maxn = max_level,
                     rin0cuts = rin0cuts,
                     init_WB = :glorot_normal, init_Wpair = :glorot_normal)

ps, st = LuxCore.setup(rng, model)

# Kill pair basis for clarity (test only ACE part)
for s in model.pairbasis.splines
   s.itp.itp.coefs[:] *= 0
end

# Convert to ETACE model
et_model = ETM.convert2et(model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

# Match parameters
et_ps.rembed.post.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]
et_ps.rembed.post.W[:, :, 2] = ps.rbasis.Wnlq[:, :, 1, 2]
et_ps.rembed.post.W[:, :, 3] = ps.rbasis.Wnlq[:, :, 2, 1]
et_ps.rembed.post.W[:, :, 4] = ps.rbasis.Wnlq[:, :, 2, 2]

et_ps.readout.W[1, :, 1] .= ps.WB[:, 1]
et_ps.readout.W[1, :, 2] .= ps.WB[:, 2]

# Get cutoff radius
rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)

# Helper to generate random structures
function rand_struct()
   sys = AtomsBuilder.bulk(:Si) * (2, 2, 1)
   rattle!(sys, 0.2u"Å")
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys
end

##

@info("Testing ETACEPotential construction")

# Create calculator from ETACE model
et_calc = ETM.ETACEPotential(et_model, et_ps, et_st, rcut)

@test et_calc.model === et_model
@test et_calc.rcut == rcut
@test et_calc.co_ps === nothing
println("ETACEPotential construction: OK")

##

@info("Testing energy consistency: ETACE model vs ETACEPotential")

for ntest = 1:20
   local sys, G, E_model, E_calc

   sys = rand_struct()
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

   # Energy from direct model evaluation
   Ei_model, _ = et_model(G, et_ps, et_st)
   E_model = sum(Ei_model)

   # Energy from calculator
   E_calc = AtomsCalculators.potential_energy(sys, et_calc)

   print_tf(@test abs(E_model - ustrip(E_calc)) < 1e-10)
end
println()

##

@info("Testing energy consistency: ETACE vs original ACE model")

# Wrap original ACE model into calculator
calc_model = M.ACEPotential(model, ps, st)

for ntest = 1:20
   local sys, E_old, E_new

   sys = rand_struct()
   E_old = AtomsCalculators.potential_energy(sys, calc_model)
   E_new = AtomsCalculators.potential_energy(sys, et_calc)

   print_tf(@test abs(ustrip(E_old) - ustrip(E_new)) < 1e-6)
end
println()

##

@info("Testing forces consistency: ETACE vs original ACE model")

for ntest = 1:20
   local sys, F_old, F_new

   sys = rand_struct()
   F_old = AtomsCalculators.forces(sys, calc_model)
   F_new = AtomsCalculators.forces(sys, et_calc)

   # Compare force magnitudes (allow small numerical differences)
   max_diff = maximum(norm(ustrip.(f1) - ustrip.(f2)) for (f1, f2) in zip(F_old, F_new))
   print_tf(@test max_diff < 1e-6)
end
println()

##

@info("Testing virial consistency: ETACE vs original ACE model")

for ntest = 1:20
   local sys, V_old, V_new

   sys = rand_struct()
   efv_old = AtomsCalculators.energy_forces_virial(sys, calc_model)
   efv_new = AtomsCalculators.energy_forces_virial(sys, et_calc)

   V_old = ustrip.(efv_old.virial)
   V_new = ustrip.(efv_new.virial)

   # Compare virial tensors
   print_tf(@test norm(V_old - V_new) / (norm(V_old) + 1e-10) < 1e-6)
end
println()

##

@info("Testing AtomsCalculators interface compliance")

sys = rand_struct()

# Test individual methods
E = AtomsCalculators.potential_energy(sys, et_calc)
F = AtomsCalculators.forces(sys, et_calc)
V = AtomsCalculators.virial(sys, et_calc)

@test E isa typeof(1.0u"eV")
@test eltype(F) <: StaticArrays.SVector
@test V isa StaticArrays.SMatrix

println("AtomsCalculators interface: OK")

##

@info("Testing combined energy_forces_virial efficiency")

sys = rand_struct()

# Combined evaluation
efv1 = AtomsCalculators.energy_forces_virial(sys, et_calc)

# Separate evaluations
E = AtomsCalculators.potential_energy(sys, et_calc)
F = AtomsCalculators.forces(sys, et_calc)
V = AtomsCalculators.virial(sys, et_calc)

@test ustrip(efv1.energy) ≈ ustrip(E)
@test all(ustrip.(efv1.forces) .≈ ustrip.(F))
@test ustrip.(efv1.virial) ≈ ustrip.(V)

println("Combined evaluation consistency: OK")

##

@info("Testing cutoff_radius function")

@test ETM.cutoff_radius(et_calc) == rcut * u"Å"
println("Cutoff radius: OK")

##

@info("Performance comparison: ETACE vs original ACE model")

# Use a fixed test structure for benchmarking
bench_sys = rand_struct()

# Warm-up runs
AtomsCalculators.energy_forces_virial(bench_sys, calc_model)
AtomsCalculators.energy_forces_virial(bench_sys, et_calc)

# Benchmark energy
t_energy_old = @belapsed AtomsCalculators.potential_energy($bench_sys, $calc_model)
t_energy_new = @belapsed AtomsCalculators.potential_energy($bench_sys, $et_calc)

# Benchmark forces
t_forces_old = @belapsed AtomsCalculators.forces($bench_sys, $calc_model)
t_forces_new = @belapsed AtomsCalculators.forces($bench_sys, $et_calc)

# Benchmark energy_forces_virial
t_efv_old = @belapsed AtomsCalculators.energy_forces_virial($bench_sys, $calc_model)
t_efv_new = @belapsed AtomsCalculators.energy_forces_virial($bench_sys, $et_calc)

println("CPU Performance comparison (times in ms):")
println("  Energy:              ACE = $(round(t_energy_old*1000, digits=3)), ETACE = $(round(t_energy_new*1000, digits=3)), ratio = $(round(t_energy_new/t_energy_old, digits=2))")
println("  Forces:              ACE = $(round(t_forces_old*1000, digits=3)), ETACE = $(round(t_forces_new*1000, digits=3)), ratio = $(round(t_forces_new/t_forces_old, digits=2))")
println("  Energy+Forces+Virial: ACE = $(round(t_efv_old*1000, digits=3)), ETACE = $(round(t_efv_new*1000, digits=3)), ratio = $(round(t_efv_new/t_efv_old, digits=2))")

##

# GPU benchmarks (if available)
# Include GPU detection utils from EquivariantTensors
et_test_utils = joinpath(dirname(dirname(pathof(ET))), "test", "test_utils")
include(joinpath(et_test_utils, "utils_gpu.jl"))

if dev !== identity
   @info("GPU Performance comparison: ETACE on GPU vs CPU")

   # NOTE: These benchmarks measure model evaluation time ONLY, with pre-constructed graphs.
   # The neighborlist/graph construction currently runs on CPU (~7ms for 250 atoms) and is
   # NOT included in the timings below. NeighbourLists.jl now has GPU support (PR #34, Dec 2025)
   # but EquivariantTensors.jl doesn't use it yet. For end-to-end GPU acceleration, the
   # neighborlist construction needs to be ported to GPU as well.

   # Use a larger system for meaningful GPU benchmark (small systems are overhead-dominated)
   # GPU kernel launch overhead is ~0.4ms, so need enough work to amortize this
   gpu_bench_sys = AtomsBuilder.bulk(:Si) * (4, 4, 4)  # 128 atoms
   rattle!(gpu_bench_sys, 0.1u"Å")
   AtomsBuilder.randz!(gpu_bench_sys, [:Si => 0.5, :O => 0.5])

   # Create graph and convert to Float32 for GPU
   G = ET.Atoms.interaction_graph(gpu_bench_sys, rcut * u"Å")
   G_32 = ET.float32(G)
   G_gpu = dev(G_32)

   et_ps_32 = ET.float32(et_ps)
   et_st_32 = ET.float32(et_st)
   et_ps_gpu = dev(et_ps_32)
   et_st_gpu = dev(et_st_32)

   # Warm-up GPU (forward pass)
   et_model(G_gpu, et_ps_gpu, et_st_gpu)

   # Benchmark GPU energy (forward pass only)
   t_energy_gpu = @belapsed begin
      Ei, _ = $et_model($G_gpu, $et_ps_gpu, $et_st_gpu)
      sum(Ei)
   end

   # Compare to CPU Float32 for fair comparison
   t_energy_cpu32 = @belapsed begin
      Ei, _ = $et_model($G_32, $et_ps_32, $et_st_32)
      sum(Ei)
   end

   println("GPU vs CPU Float32 comparison ($(length(gpu_bench_sys)) atoms, $(length(G.ii)) edges):")
   println("  Energy:    CPU = $(round(t_energy_cpu32*1000, digits=3))ms, GPU = $(round(t_energy_gpu*1000, digits=3))ms, speedup = $(round(t_energy_cpu32/t_energy_gpu, digits=1))x")

   # Try GPU gradients (may not be supported yet - gradients w.r.t. positions
   # require Zygote through P4ML which has GPU compat issues; see ET test_ace_ka.jl:196-197)
   gpu_grads_work = try
      ETM.site_grads(et_model, G_gpu, et_ps_gpu, et_st_gpu)
      true
   catch e
      @warn("GPU position gradients not yet supported (needed for forces): $(typeof(e).name.name)")
      false
   end

   if gpu_grads_work
      # Benchmark GPU gradients (for forces)
      t_grads_gpu = @belapsed ETM.site_grads($et_model, $G_gpu, $et_ps_gpu, $et_st_gpu)
      t_grads_cpu32 = @belapsed ETM.site_grads($et_model, $G_32, $et_ps_32, $et_st_32)
      println("  Gradients: CPU = $(round(t_grads_cpu32*1000, digits=3)), GPU = $(round(t_grads_gpu*1000, digits=3)), speedup = $(round(t_grads_cpu32/t_grads_gpu, digits=2))x")
   else
      println("  Gradients: Skipped (GPU gradients not yet supported)")
   end
else
   @info("No GPU available, skipping GPU benchmarks")
end

##

@info("All Phase 1 tests passed!")

# ============================================================================
#  Phase 2 Tests: SiteEnergyModel Interface, WrappedSiteCalculator, StackedCalculator
# ============================================================================

@info("Testing Phase 2: SiteEnergyModel interface and calculators")

##

@info("Testing E0Model")

# Create E0 model with reference energies
E0_Si = -0.846
E0_O = -2.15
E0 = ETM.E0Model(Dict(:Si => E0_Si, :O => E0_O))

# Test cutoff radius
@test ETM.cutoff_radius(E0) == 0.0
println("E0Model cutoff_radius: OK")

# Test site energies
sys = rand_struct()
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
Ei_E0 = ETM.site_energies(E0, G, nothing, nothing)

# Count Si and O atoms
n_Si = count(node -> node.z == AtomsBase.ChemicalSpecies(:Si), G.node_data)
n_O = count(node -> node.z == AtomsBase.ChemicalSpecies(:O), G.node_data)
expected_E0 = n_Si * E0_Si + n_O * E0_O

@test length(Ei_E0) == length(sys)
@test sum(Ei_E0) ≈ expected_E0
println("E0Model site_energies: OK")

# Test site energy gradients (should be zero)
∂G_E0 = ETM.site_energy_grads(E0, G, nothing, nothing)
@test all(norm(e.𝐫) == 0 for e in ∂G_E0.edge_data)
println("E0Model site_energy_grads (zero): OK")

##

@info("Testing WrappedETACE")

# Create wrapped ETACE model
wrapped_ace = ETM.WrappedETACE(et_model, et_ps, et_st, rcut)

# Test cutoff radius
@test ETM.cutoff_radius(wrapped_ace) == rcut
println("WrappedETACE cutoff_radius: OK")

# Test site energies match direct evaluation
Ei_wrapped = ETM.site_energies(wrapped_ace, G, nothing, nothing)
Ei_direct, _ = et_model(G, et_ps, et_st)
@test Ei_wrapped ≈ Ei_direct
println("WrappedETACE site_energies: OK")

# Test site energy gradients match direct evaluation
∂G_wrapped = ETM.site_energy_grads(wrapped_ace, G, nothing, nothing)
∂G_direct = ETM.site_grads(et_model, G, et_ps, et_st)
@test all(∂G_wrapped.edge_data[i].𝐫 ≈ ∂G_direct.edge_data[i].𝐫 for i in 1:length(G.edge_data))
println("WrappedETACE site_energy_grads: OK")

##

@info("Testing WrappedSiteCalculator")

# Wrap E0 model in a calculator
E0_calc = ETM.WrappedSiteCalculator(E0)
@test ustrip(u"Å", ETM.cutoff_radius(E0_calc)) == 3.0  # minimum cutoff
println("WrappedSiteCalculator(E0) cutoff_radius: OK")

# Wrap ETACE model in a calculator
ace_site_calc = ETM.WrappedSiteCalculator(wrapped_ace)
@test ustrip(u"Å", ETM.cutoff_radius(ace_site_calc)) == rcut
println("WrappedSiteCalculator(ETACE) cutoff_radius: OK")

# Test E0 calculator energy
sys = rand_struct()
E_E0_calc = AtomsCalculators.potential_energy(sys, E0_calc)
G = ET.Atoms.interaction_graph(sys, 3.0 * u"Å")
n_Si = count(node -> node.z == AtomsBase.ChemicalSpecies(:Si), G.node_data)
n_O = count(node -> node.z == AtomsBase.ChemicalSpecies(:O), G.node_data)
expected_E = (n_Si * E0_Si + n_O * E0_O) * u"eV"
@test ustrip(E_E0_calc) ≈ ustrip(expected_E)
println("WrappedSiteCalculator(E0) energy: OK")

# Test E0 calculator forces (should be zero)
F_E0_calc = AtomsCalculators.forces(sys, E0_calc)
@test all(norm(ustrip.(f)) < 1e-14 for f in F_E0_calc)
println("WrappedSiteCalculator(E0) forces (zero): OK")

# Test ETACE calculator matches ETACEPotential
sys = rand_struct()
E_ace_site = AtomsCalculators.potential_energy(sys, ace_site_calc)
E_ace_pot = AtomsCalculators.potential_energy(sys, et_calc)
@test ustrip(E_ace_site) ≈ ustrip(E_ace_pot)
println("WrappedSiteCalculator(ETACE) energy matches ETACEPotential: OK")

F_ace_site = AtomsCalculators.forces(sys, ace_site_calc)
F_ace_pot = AtomsCalculators.forces(sys, et_calc)
max_diff = maximum(norm(ustrip.(f1) - ustrip.(f2)) for (f1, f2) in zip(F_ace_site, F_ace_pot))
@test max_diff < 1e-10
println("WrappedSiteCalculator(ETACE) forces match ETACEPotential: OK")

##

@info("Testing StackedCalculator construction")

# Create stacked calculator with E0 + ACE (both wrapped)
stacked = ETM.StackedCalculator((E0_calc, ace_site_calc))

@test ustrip(u"Å", ETM.cutoff_radius(stacked)) == rcut
@test length(stacked.calcs) == 2
println("StackedCalculator construction: OK")

##

@info("Testing StackedCalculator energy consistency")

for ntest = 1:10
   local sys, E_stacked, E_separate

   sys = rand_struct()

   # Energy from stacked calculator
   E_stacked = AtomsCalculators.potential_energy(sys, stacked)

   # Energy from separate evaluations
   E_E0 = AtomsCalculators.potential_energy(sys, E0_calc)
   E_ace = AtomsCalculators.potential_energy(sys, ace_site_calc)
   E_separate = E_E0 + E_ace

   print_tf(@test ustrip(E_stacked) ≈ ustrip(E_separate))
end
println()

##

@info("Testing StackedCalculator forces consistency")

for ntest = 1:10
   local sys, F_stacked, F_ace_only, max_diff

   sys = rand_struct()

   # Forces from stacked calculator
   F_stacked = AtomsCalculators.forces(sys, stacked)

   # Forces from ACE-only (E0 has zero forces)
   F_ace_only = AtomsCalculators.forces(sys, et_calc)

   # Should be identical since E0 contributes zero forces
   max_diff = maximum(norm(ustrip.(f1) - ustrip.(f2)) for (f1, f2) in zip(F_stacked, F_ace_only))
   print_tf(@test max_diff < 1e-10)
end
println()

##

@info("Testing StackedCalculator virial consistency")

for ntest = 1:10
   local sys, efv_stacked, efv_ace_only

   sys = rand_struct()

   efv_stacked = AtomsCalculators.energy_forces_virial(sys, stacked)
   efv_ace_only = AtomsCalculators.energy_forces_virial(sys, et_calc)

   # Virial should match (E0 has zero virial)
   V_stacked = ustrip.(efv_stacked.virial)
   V_ace_only = ustrip.(efv_ace_only.virial)

   print_tf(@test norm(V_stacked - V_ace_only) / (norm(V_ace_only) + 1e-10) < 1e-10)
end
println()

##

@info("Testing StackedCalculator with E0 only")

# Create stacked calculator with just E0
E0_only_stacked = ETM.StackedCalculator((E0_calc,))

sys = rand_struct()
E = AtomsCalculators.potential_energy(sys, E0_only_stacked)
F = AtomsCalculators.forces(sys, E0_only_stacked)

# Energy should match E0_calc
E_direct = AtomsCalculators.potential_energy(sys, E0_calc)
@test ustrip(E) ≈ ustrip(E_direct)
println("StackedCalculator(E0 only) energy: OK")

# Forces should be zero
@test all(norm(ustrip.(f)) < 1e-14 for f in F)
println("StackedCalculator(E0 only) forces (zero): OK")

##

@info("All Phase 2 tests passed!")

## ============================================================================
##  Phase 5: Training Assembly Tests
## ============================================================================

@info("Testing Phase 5: Training assembly functions")

##

@info("Testing length_basis")
nparams = ETM.length_basis(et_calc)
nbasis = et_model.readout.in_dim
nspecies = et_model.readout.ncat
@test nparams == nbasis * nspecies
println("length_basis: OK (nparams=$nparams, nbasis=$nbasis, nspecies=$nspecies)")

##

@info("Testing get/set_linear_parameters round-trip")
θ_orig = ETM.get_linear_parameters(et_calc)
@test length(θ_orig) == nparams

# Modify and restore
θ_test = randn(nparams)
ETM.set_linear_parameters!(et_calc, θ_test)
θ_check = ETM.get_linear_parameters(et_calc)
@test θ_check ≈ θ_test

# Restore original
ETM.set_linear_parameters!(et_calc, θ_orig)
@test ETM.get_linear_parameters(et_calc) ≈ θ_orig
println("get/set_linear_parameters round-trip: OK")

##

@info("Testing potential_energy_basis")
sys = rand_struct()
E_basis = ETM.potential_energy_basis(sys, et_calc)
@test length(E_basis) == nparams
@test eltype(ustrip.(E_basis)) <: Real
println("potential_energy_basis shape: OK")

##

@info("Testing energy_forces_virial_basis")
efv_basis = ETM.energy_forces_virial_basis(sys, et_calc)
natoms = length(sys)

@test length(efv_basis.energy) == nparams
@test size(efv_basis.forces) == (natoms, nparams)
@test length(efv_basis.virial) == nparams
println("energy_forces_virial_basis shapes: OK")

##

@info("Testing linear combination gives correct energy")

# E = dot(E_basis, θ) should match potential_energy
θ = ETM.get_linear_parameters(et_calc)
E_from_basis = dot(ustrip.(efv_basis.energy), θ)
E_direct = ustrip(u"eV", AtomsCalculators.potential_energy(sys, et_calc))

print_tf(@test E_from_basis ≈ E_direct rtol=1e-10)
println()
println("Energy from basis: OK")

##

@info("Testing linear combination gives correct forces")

# F = efv_basis.forces * θ should match forces
F_from_basis = efv_basis.forces * θ
F_direct = AtomsCalculators.forces(sys, et_calc)

max_diff = maximum(norm(ustrip.(f1) - ustrip.(f2)) for (f1, f2) in zip(F_from_basis, F_direct))
print_tf(@test max_diff < 1e-10)
println()
println("Forces from basis: OK (max_diff = $max_diff)")

##

@info("Testing linear combination gives correct virial")

# V = sum(θ .* virial) should match virial
V_from_basis = sum(θ[i] * ustrip.(efv_basis.virial[i]) for i in 1:nparams)
V_direct = ustrip.(AtomsCalculators.virial(sys, et_calc))

virial_diff = maximum(abs.(V_from_basis - V_direct))
print_tf(@test virial_diff < 1e-10)
println()
println("Virial from basis: OK (max_diff = $virial_diff)")

##

@info("Testing potential_energy_basis matches energy from efv_basis")
@test ustrip.(E_basis) ≈ ustrip.(efv_basis.energy)
println("potential_energy_basis consistency: OK")

##

@info("All Phase 5 tests passed!")
