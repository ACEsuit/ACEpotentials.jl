# Integration test for ETACE calculators
#
# This test verifies that ETACE calculators produce comparable results
# to the original ACE models when used for evaluation (not fitting).
#
# Note: convert2et only supports LearnableRnlrzzBasis (not SplineRnlrzzBasis),
# so we use ace_model() directly instead of ace1_model().

using Test
using ACEpotentials
M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

using ExtXYZ, AtomsBase, Unitful, StaticArrays
using AtomsCalculators
using LazyArtifacts
using LuxCore, Lux, Random, LinearAlgebra

@info("ETACE Integration Test: Silicon dataset")

## ----- setup -----

# Build model using ace_model (LearnableRnlrzzBasis, compatible with convert2et)
elements = (:Si,)
level = M.TotalDegree()
max_level = 12
order = 3
maxl = 6

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)

rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(; elements = elements, order = order,
                         Ytype = :solid, level = level, max_level = max_level,
                         maxl = maxl, pair_maxn = max_level,
                         rin0cuts = rin0cuts,
                         init_WB = :glorot_normal, init_Wpair = :glorot_normal)

ps, st = Lux.setup(rng, ace_model)

# Create ACE calculator
model = M.ACEPotential(ace_model, ps, st)

# Load dataset
data = ExtXYZ.load(artifact"Si_tiny_dataset" * "/Si_tiny.xyz")

data_keys = [:energy_key => "dft_energy",
             :force_key  => "dft_force",
             :virial_key => "dft_virial"]

weights = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0),
               "liq" => Dict("E"=>10.0, "F"=>0.66, "V"=>0.25))

## ----- Fit original ACE model -----

@info("Fitting original ACE model with QR solver")
acefit!(data, model;
        data_keys...,
        weights = weights,
        solver = ACEfit.QR())

ace_err = ACEpotentials.compute_errors(data, model; data_keys..., weights=weights)
@info("Original ACE RMSE (set):",
      E=ace_err["rmse"]["set"]["E"],
      F=ace_err["rmse"]["set"]["F"],
      V=ace_err["rmse"]["set"]["V"])

# Store for comparison
ace_rmse_E = ace_err["rmse"]["set"]["E"]
ace_rmse_F = ace_err["rmse"]["set"]["F"]
ace_rmse_V = ace_err["rmse"]["set"]["V"]

## ----- Convert to ETACE and compare -----

@info("Converting to ETACE model")

# Update ps from model after fitting
ps = model.ps
st = model.st

# Convert to ETACE
et_model = ETM.convert2et(ace_model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

# Copy radial basis parameters (single species case)
et_ps.rembed.post.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]

# Copy readout parameters
et_ps.readout.W[1, :, 1] .= ps.WB[:, 1]

# Get cutoff
rcut = maximum(a.rcut for a in ace_model.pairbasis.rin0cuts)

# Create ETACEPotential
et_calc = ETM.ETACEPotential(et_model, et_ps, et_st, rcut)

## ----- Test energy consistency -----

@info("Testing energy consistency between ACE and ETACE")

# Skip isolated atom (index 1) - ETACE requires at least 2 atoms for graph construction
max_energy_diff = 0.0
for (i, sys) in enumerate(data[2:min(11, length(data))])
    local E_ace = ustrip(u"eV", AtomsCalculators.potential_energy(sys, model))
    local E_etace = ustrip(u"eV", AtomsCalculators.potential_energy(sys, et_calc))
    local diff = abs(E_ace - E_etace)
    max_energy_diff = max(max_energy_diff, diff)
end

@info("Max energy difference: $max_energy_diff eV")
@test max_energy_diff < 1e-10
println("Energy consistency: OK (max_diff = $max_energy_diff eV)")

## ----- Test forces consistency -----

@info("Testing forces consistency between ACE and ETACE")

max_force_diff = 0.0
for (i, sys) in enumerate(data[1:min(10, length(data))])
    F_ace = AtomsCalculators.forces(sys, model)
    F_etace = AtomsCalculators.forces(sys, et_calc)
    for (f1, f2) in zip(F_ace, F_etace)
        diff = norm(ustrip.(f1) - ustrip.(f2))
        max_force_diff = max(max_force_diff, diff)
    end
end

@info("Max force difference: $max_force_diff eV/Å")
@test max_force_diff < 1e-10
println("Forces consistency: OK (max_diff = $max_force_diff eV/Å)")

## ----- Test virial consistency -----

@info("Testing virial consistency between ACE and ETACE")

max_virial_diff = 0.0
for (i, sys) in enumerate(data[1:min(10, length(data))])
    V_ace = AtomsCalculators.virial(sys, model)
    V_etace = AtomsCalculators.virial(sys, et_calc)
    diff = maximum(abs.(ustrip.(V_ace) - ustrip.(V_etace)))
    max_virial_diff = max(max_virial_diff, diff)
end

@info("Max virial difference: $max_virial_diff eV")
@test max_virial_diff < 1e-9
println("Virial consistency: OK (max_diff = $max_virial_diff eV)")

## ----- Test training basis assembly -----

@info("Testing training basis assembly")

# Pick a test structure
sys = data[5]
natoms = length(sys)
nparams = ETM.length_basis(et_calc)

# Get basis
efv_basis = ETM.energy_forces_virial_basis(sys, et_calc)

# Verify shapes
@test length(efv_basis.energy) == nparams
@test size(efv_basis.forces) == (natoms, nparams)
@test length(efv_basis.virial) == nparams

# Verify linear combination matches direct evaluation
θ = ETM.get_linear_parameters(et_calc)

E_from_basis = dot(ustrip.(efv_basis.energy), θ)
E_direct = ustrip(u"eV", AtomsCalculators.potential_energy(sys, et_calc))
@test isapprox(E_from_basis, E_direct, rtol=1e-10)

F_from_basis = efv_basis.forces * θ
F_direct = AtomsCalculators.forces(sys, et_calc)
max_F_diff = maximum(norm(ustrip.(f1) - ustrip.(f2)) for (f1, f2) in zip(F_from_basis, F_direct))
@test max_F_diff < 1e-10

V_from_basis = sum(θ[k] * ustrip.(efv_basis.virial[k]) for k in 1:nparams)
V_direct = ustrip.(AtomsCalculators.virial(sys, et_calc))
max_V_diff = maximum(abs.(V_from_basis - V_direct))
@test max_V_diff < 1e-9

println("Training basis assembly: OK")

## ----- Test StackedCalculator with E0 -----

@info("Testing StackedCalculator with E0Model")

# Create E0 model with arbitrary E0 value for testing
E0s = Dict(14 => -158.54496821)  # Si atomic number => E0
E0_model = ETM.E0Model(E0s)
E0_calc = ETM.WrappedSiteCalculator(E0_model)

# Create wrapped ETACE
wrapped_etace = ETM.WrappedETACE(et_model, et_ps, et_st, rcut)
ace_calc = ETM.WrappedSiteCalculator(wrapped_etace)

# Stack them
stacked = ETM.StackedCalculator((E0_calc, ace_calc))

# Test on a few structures
for (i, sys) in enumerate(data[1:5])
    E_E0 = AtomsCalculators.potential_energy(sys, E0_calc)
    E_ace = AtomsCalculators.potential_energy(sys, ace_calc)
    E_stacked = AtomsCalculators.potential_energy(sys, stacked)

    expected = ustrip(E_E0) + ustrip(E_ace)
    actual = ustrip(E_stacked)

    @test isapprox(expected, actual, rtol=1e-10)
end

println("StackedCalculator: OK")

## ----- Summary -----

@info("All ETACE integration tests passed!")
@info("Summary:")
@info("  - Energy matches original ACE to < 1e-10 eV")
@info("  - Forces match original ACE to < 1e-10 eV/Å")
@info("  - Virial matches original ACE to < 1e-9 eV")
@info("  - Training basis assembly verified")
@info("  - StackedCalculator composition verified")
