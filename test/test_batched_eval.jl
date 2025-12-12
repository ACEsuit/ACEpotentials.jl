# Test batched evaluation for multiple structures

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

# Build unified calculator (supports both energy/forces/virial and basis evaluation)
calc_et = M.build_et_calculator(model, ps, st)

function rand_struct(n_repeat=(2,1,1))
   sys = AtomsBuilder.bulk(:Si) * n_repeat
   rattle!(sys, 0.2u"Å")
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys
end

##

println("\nTesting BatchedETGraph construction...")

@testset "BatchedETGraph" begin

   @testset "batch_graphs construction" begin
      println("\n  Testing batch_graphs...")

      # Create test structures of different sizes
      sys1 = rand_struct((2,1,1))
      sys2 = rand_struct((2,2,1))
      sys3 = rand_struct((3,1,1))

      systems = [sys1, sys2, sys3]
      rcut = 5.5u"Å"

      bg = M.batch_graphs(systems, rcut)

      @test bg.n_structures == 3
      @test length(bg.atom_offsets) == 4
      @test length(bg.edge_offsets) == 4

      # Check atom counts
      @test bg.n_atoms_per_structure[1] == length(sys1)
      @test bg.n_atoms_per_structure[2] == length(sys2)
      @test bg.n_atoms_per_structure[3] == length(sys3)

      # Check total counts
      @test M.total_atoms(bg) == sum(length.(systems))
      @test bg.atom_offsets[end] == sum(length.(systems))

      println("    Batch: ", bg)
   end

   @testset "individual graph equivalence" begin
      println("\n  Testing that batched graph contains correct data...")

      sys1 = rand_struct((2,1,1))
      sys2 = rand_struct((2,2,1))
      systems = [sys1, sys2]
      rcut = 5.5u"Å"

      # Build individual graphs
      g1 = ET.Atoms.interaction_graph(sys1, rcut)
      g2 = ET.Atoms.interaction_graph(sys2, rcut)

      # Build batched graph
      bg = M.batch_graphs(systems, rcut)

      # Check that structure 1 edges are preserved
      edge_range_1 = M.get_structure_edges(bg, 1)
      @test length(edge_range_1) == ET.nedges(g1)

      # Check that structure 2 edges are preserved
      edge_range_2 = M.get_structure_edges(bg, 2)
      @test length(edge_range_2) == ET.nedges(g2)

      # Node data should be concatenated
      atom_range_1 = M.get_structure_atoms(bg, 1)
      atom_range_2 = M.get_structure_atoms(bg, 2)
      @test length(atom_range_1) == length(sys1)
      @test length(atom_range_2) == length(sys2)
   end

end

##

println("\nTesting batched energy evaluation...")

@testset "Batched Energy Evaluation" begin

   @testset "energies match individual" begin
      println("\n  Testing batched energies match individual...")

      for ntest = 1:5
         # Create random structures
         sys1 = rand_struct((2,1,1))
         sys2 = rand_struct((2,2,1))
         sys3 = rand_struct((3,1,1))
         systems = [sys1, sys2, sys3]

         # Individual energies
         E1 = ustrip(AtomsCalculators.potential_energy(sys1, calc_et))
         E2 = ustrip(AtomsCalculators.potential_energy(sys2, calc_et))
         E3 = ustrip(AtomsCalculators.potential_energy(sys3, calc_et))
         E_individual = [E1, E2, E3]

         # Batched energies
         E_batched = ustrip.(M.evaluate_batched_energies(calc_et, systems))

         # Compare
         E_err = maximum(abs.(E_batched - E_individual))
         print_tf(@test E_err < 1e-10)
      end
      println()
   end

end

##

println("\nTesting batched basis evaluation...")

@testset "Batched Basis Evaluation" begin

   @testset "basis matches individual" begin
      println("\n  Testing batched basis matches individual...")

      for ntest = 1:3
         sys1 = rand_struct((2,1,1))
         sys2 = rand_struct((2,2,1))
         systems = [sys1, sys2]

         # Individual basis evaluation (unified calculator supports both energy and basis)
         B1 = M.evaluate_basis(calc_et, sys1)
         B2 = M.evaluate_basis(calc_et, sys2)

         # Batched basis evaluation
         B_batched = M.evaluate_batched_basis(calc_et, systems)

         # Compare
         B1_err = maximum(abs.(B_batched[1] - B1))
         B2_err = maximum(abs.(B_batched[2] - B2))

         print_tf(@test B1_err < 1e-10)
         print_tf(@test B2_err < 1e-10)
      end
      println()
   end

end

##

println("\nTesting batched forces/virial evaluation...")

@testset "Batched EFV Evaluation" begin

   @testset "efv matches individual" begin
      println("\n  Testing batched EFV matches individual...")

      for ntest = 1:3
         sys1 = rand_struct((2,1,1))
         sys2 = rand_struct((2,2,1))
         systems = [sys1, sys2]

         # Individual EFV
         efv1 = AtomsCalculators.energy_forces_virial(sys1, calc_et)
         efv2 = AtomsCalculators.energy_forces_virial(sys2, calc_et)

         # Batched EFV
         efv_batched = M.evaluate_batched_efv(calc_et, systems)

         # Compare energies
         E1_err = abs(ustrip(efv_batched[1].energy) - ustrip(efv1.energy))
         E2_err = abs(ustrip(efv_batched[2].energy) - ustrip(efv2.energy))
         print_tf(@test E1_err < 1e-10)
         print_tf(@test E2_err < 1e-10)

         # Compare forces
         F1_err = maximum(norm.(ustrip.(efv_batched[1].forces) .- ustrip.(efv1.forces)))
         F2_err = maximum(norm.(ustrip.(efv_batched[2].forces) .- ustrip.(efv2.forces)))
         print_tf(@test F1_err < 1e-10)
         print_tf(@test F2_err < 1e-10)

         # Compare virials
         V1_err = maximum(abs.(ustrip(efv_batched[1].virial) .- ustrip(efv1.virial)))
         V2_err = maximum(abs.(ustrip(efv_batched[2].virial) .- ustrip(efv2.virial)))
         print_tf(@test V1_err < 1e-10)
         print_tf(@test V2_err < 1e-10)
      end
      println()
   end

end

println("\nAll batched evaluation tests completed!")
