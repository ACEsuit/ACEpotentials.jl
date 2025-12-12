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

# Build the ET model manually (same as in new_backend.jl)
rbasis = model.rbasis
et_i2z = AtomsBase.ChemicalSpecies.(rbasis._i2z)
et_rbasis = M._convert_Rnl_learnable(rbasis; zlist = et_i2z, rfun = x -> norm(x.𝐫))
et_rspec = rbasis.spec

et_ybasis = Lux.Chain(𝐫ij = ET.NTtransform(x -> x.𝐫), Y = model.ybasis)
et_yspec = P4ML.natural_indices(et_ybasis.layers.Y)

et_embed = ET.EdgeEmbed(Lux.BranchLayer(; Rnl = et_rbasis, Ylm = et_ybasis))

AA_spec = model.tensor.meta["𝔸spec"]
et_mb_spec = unique([[(n=b.n, l=b.l) for b in bb] for bb in AA_spec])

et_mb_basis = ET.sparse_equivariant_tensor(
      L = 0,
      mb_spec = et_mb_spec,
      Rnl_spec = et_rspec,
      Ylm_spec = et_yspec,
      basis = real)

et_readout = let zlist = et_i2z
      __zi = x -> ET.cat2idx(zlist, x.s)
      ET.SelectLinL(et_mb_basis.lens[1], 1, length(et_i2z), __zi)
end

et_basis = Lux.Chain(;
            embed = et_embed,
            ace = et_mb_basis,
            unwrp = Lux.WrappedFunction(x -> x[1]))

et_model = Lux.Chain(
      L1 = Lux.BranchLayer(;
         basis = et_basis,
         nodes = Lux.WrappedFunction(G -> G.node_data)),
      Ei = et_readout,
      E = Lux.WrappedFunction(sum))

et_ps, et_st = LuxCore.setup(rng, et_model)

# Copy parameters
NZ = length(et_i2z)
for i in 1:NZ, j in 1:NZ
   idx = (i-1)*NZ + j
   et_ps.L1.basis.embed.Rnl.connection.W[:, :, idx] = ps.rbasis.Wnlq[:, :, i, j]
end

et_ps.Ei.W[1, :, 1] .= ps.WB[:, 1]
et_ps.Ei.W[1, :, 2] .= ps.WB[:, 2]

##

# Create calculators
calc_ref = ACEpotentials.ACEPotential(model, ps, st)
rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)

# Create ETCalculator directly
calc_et = M.ETCalculator(et_model, et_ps, et_st, rcut * u"Å")

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

end

println("\nAll tests passed!")
