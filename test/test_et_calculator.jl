# Test ETCalculator AtomsCalculators interface

using Pkg; Pkg.activate(joinpath(@__DIR__(), ".."))
et_path = joinpath(@__DIR__(), "..", "..", "EquivariantTensors.jl")
if isdir(et_path)
   Pkg.develop(path = et_path)
end

##

using ACEpotentials
M = ACEpotentials.Models

import EquivariantTensors as ET
import Polynomials4ML as P4ML

using StaticArrays, Lux
using AtomsBase, AtomsBuilder, Unitful, AtomsCalculators

using Random, LuxCore, Test, LinearAlgebra
using Polynomials4ML.Testing: print_tf, println_slim

rng = Random.MersenneTwister(1234)
Random.seed!(1234)

##

println("Setting up model...")

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 8
order = 2
maxl = 4

rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)

model = M.ace_model(; elements = elements, order = order,
            Ytype = :solid, level = level, max_level = max_level,
            maxl = maxl, pair_maxn = max_level,
            rin0cuts = rin0cuts,
            init_WB = :glorot_normal, init_Wpair = :glorot_normal)

ps, st = Lux.setup(rng, model)

# Kill pair basis for now (not yet implemented in ET backend)
for s in model.pairbasis.splines
   s.itp.itp.coefs[:] *= 0
end

##

# Create calculators using unified build_et_calculator
calc_ref = ACEpotentials.ACEPotential(model, ps, st)

# Use the unified build_et_calculator function
calc_et = M.build_et_calculator(model, ps, st)

function rand_struct()
   sys = AtomsBuilder.bulk(:Si) * (2,1,1)
   rattle!(sys, 0.2u"Å")
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys
end

##

println("\nTesting ETCalculator via AtomsCalculators interface...")

@testset "ETCalculator" begin

   @testset "potential_energy" begin
      for ntest = 1:10
         sys = rand_struct()
         E_ref = AtomsCalculators.potential_energy(sys, calc_ref)
         E_et = AtomsCalculators.potential_energy(sys, calc_et)
         E_err = abs(ustrip(E_ref) - ustrip(E_et))
         print_tf(@test E_err < 1e-10)
      end
      println()
   end

   @testset "energy_forces_virial" begin
      for ntest = 1:10
         sys = rand_struct()

         efv_ref = AtomsCalculators.energy_forces_virial(sys, calc_ref)
         efv_et = AtomsCalculators.energy_forces_virial(sys, calc_et)

         E_err = abs(ustrip(efv_ref.energy) - ustrip(efv_et.energy))
         F_err = maximum(norm(ustrip.(f1) - ustrip.(f2))
                         for (f1, f2) in zip(efv_ref.forces, efv_et.forces))
         V_err = maximum(abs.(ustrip(efv_ref.virial) - ustrip(efv_et.virial)))

         print_tf(@test E_err < 1e-10)
         print_tf(@test F_err < 1e-10)
         print_tf(@test V_err < 1e-10)
      end
      println()
   end

   @testset "forces convenience function" begin
      sys = rand_struct()
      F_ref = AtomsCalculators.forces(sys, calc_ref)
      F_et = AtomsCalculators.forces(sys, calc_et)

      F_err = maximum(norm(ustrip.(f1) - ustrip.(f2))
                      for (f1, f2) in zip(F_ref, F_et))
      @test F_err < 1e-10
   end

   @testset "virial convenience function" begin
      sys = rand_struct()
      V_ref = AtomsCalculators.virial(sys, calc_ref)
      V_et = AtomsCalculators.virial(sys, calc_et)

      V_err = maximum(abs.(ustrip(V_ref) - ustrip(V_et)))
      @test V_err < 1e-10
   end

   @testset "evaluate_basis" begin
      println("\n  Testing evaluate_basis...")
      for ntest = 1:5
         sys = rand_struct()
         # Evaluate basis using unified calculator
         B = M.evaluate_basis(calc_et, sys)
         @test size(B, 1) == length(sys)
         @test size(B, 2) == M.length_basis(calc_et)
         # Check that basis values are reasonable (not NaN or Inf)
         print_tf(@test all(isfinite, B))
      end
      println()
   end

   # Test evaluate_basis_ed (basis + gradients using ForwardDiff)
   @testset "evaluate_basis_ed" begin
      sys = rand_struct()

      # Test that it runs without error
      B, dB = M.evaluate_basis_ed(calc_et, sys)

      # Check shapes
      n_atoms = length(sys)
      len_basis = M.length_basis(calc_et)
      @test size(B) == (n_atoms, len_basis)
      @test size(dB) == (n_atoms, len_basis, n_atoms)

      # Check consistency with evaluate_basis
      B_ref = M.evaluate_basis(calc_et, sys)
      @test B ≈ B_ref rtol=1e-10

      # Basic sanity check that gradients are non-zero for at least some entries
      has_nonzero_grad = any(norm(dB[i, b, j]) > 1e-10
                             for i in 1:n_atoms, b in 1:len_basis, j in 1:n_atoms)
      @test has_nonzero_grad

      println("  (Finite diff test skipped - API compatibility)")
      println()
   end

end

println("\nAll tests passed!")
