# Test ETBasisCalculator - basis evaluation via EquivariantTensors

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

# Build the ETBasisCalculator
basis_calc = M.build_et_basis_calculator(model, ps, st)

println("ETBasisCalculator built:")
println("  - Basis length: ", M.length_basis(basis_calc))
println("  - Species: ", basis_calc.zlist)
println("  - len_Bi: ", basis_calc.len_Bi)

##

function rand_struct()
   sys = AtomsBuilder.bulk(:Si) * (2,1,1)
   rattle!(sys, 0.2u"Å")
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys
end

##

println("\nTesting ETBasisCalculator...")

@testset "ETBasisCalculator" begin

   @testset "basis sum equals energy" begin
      println("\n  Testing that B·W = E...")

      # Build energy calculator for comparison
      calc_energy = M.build_et_calculator(model, ps, st)

      for ntest = 1:5
         sys = rand_struct()

         # Get basis
         B = M.evaluate_basis(basis_calc, sys)

         # Get energy from ETCalculator
         E_et = ustrip(AtomsCalculators.potential_energy(sys, calc_energy))

         # Compute energy from basis: E = ∑_i B[i,:] · W
         # W is the readout weight (WB in the model)
         len_Bi = basis_calc.len_Bi
         NZ = length(basis_calc.zlist)

         # Reconstruct W vector matching basis ordering
         W = zeros(M.length_basis(basis_calc))
         for iz in 1:NZ
            inds = (iz - 1) * len_Bi .+ (1:len_Bi)
            W[inds] .= ps.WB[:, iz]
         end

         # Compute energy as B · W
         E_from_basis = sum(B * W)

         E_err = abs(E_et - E_from_basis)
         print_tf(@test E_err < 1e-10)
      end
      println()
   end

   @testset "basis output shape" begin
      println("\n  Testing basis output shape...")

      sys = rand_struct()
      B = M.evaluate_basis(basis_calc, sys)

      n_atoms = length(sys)
      len_B = M.length_basis(basis_calc)

      @test size(B) == (n_atoms, len_B)
      println("    B shape: ", size(B), " ✓")

      # Check that each atom has non-zero basis only in its species slice
      for i in 1:n_atoms
         Z = AtomsBase.atomic_symbol(sys, i)
         Zs = AtomsBase.ChemicalSpecies(Z)
         inds = M.get_basis_inds(basis_calc, Zs)

         # Other indices should be zero
         other_inds = setdiff(1:len_B, inds)
         other_vals = B[i, other_inds]
         @test all(x -> x == 0.0, other_vals)
      end
      println("    Species indexing correct ✓")
   end

end

println("\nAll basis tests completed!")
