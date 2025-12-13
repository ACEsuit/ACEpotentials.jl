# Test forces and virial via Zygote through EquivariantTensors backend
# This is a standalone test that can be run to verify the gradient approach

using Pkg; Pkg.activate(joinpath(@__DIR__(), ".."))
# Use local EquivariantTensors if available
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
using Zygote

# Load shared test utilities
include("test_utils.jl")

rng = Random.MersenneTwister(1234)
Random.seed!(1234)

##

println("Setting up model...")

# Use shared test model setup
model, ps, st, _ = setup_test_model(; rng=rng)

##
# Build ET model components

rbasis = model.rbasis
et_i2z = AtomsBase.ChemicalSpecies.(rbasis._i2z)
et_rbasis = M._convert_Rnl_learnable(rbasis; zlist = et_i2z,
                                        rfun = x -> norm(x.𝐫))

et_rspec = rbasis.spec

et_ybasis = Chain(𝐫ij = ET.NTtransform(x -> x.𝐫),
                   Y = model.ybasis)
et_yspec = P4ML.natural_indices(et_ybasis.layers.Y)

et_embed = ET.EdgeEmbed(BranchLayer(;
               Rnl = et_rbasis,
               Ylm = et_ybasis))

AA_spec = model.tensor.meta["𝔸spec"]
et_mb_spec = unique([[(n=b.n, l=b.l) for b in bb] for bb in AA_spec])

et_mb_basis = ET.sparse_equivariant_tensor(
      L = 0,
      mb_spec = et_mb_spec,
      Rnl_spec = et_rspec,
      Ylm_spec = et_yspec,
      basis = real
   )

et_readout_2 = let zlist = et_i2z
      __zi = x -> ET.cat2idx(zlist, x.s)
      ET.SelectLinL(
               et_mb_basis.lens[1],
               1,
               length(et_i2z),
               __zi)
end

et_basis = Lux.Chain(;
            embed = et_embed,
              ace = et_mb_basis,
            unwrp = WrappedFunction(x -> x[1]),
            )

et_model = Lux.Chain(
      L1 = Lux.BranchLayer(;
         basis = et_basis,
         nodes = WrappedFunction(G -> G.node_data),
         ),
      Ei = et_readout_2,
      E = WrappedFunction(sum),
  )
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

# Copy parameters
et_ps.L1.basis.embed.Rnl.connection.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]
et_ps.L1.basis.embed.Rnl.connection.W[:, :, 2] = ps.rbasis.Wnlq[:, :, 1, 2]
et_ps.L1.basis.embed.Rnl.connection.W[:, :, 3] = ps.rbasis.Wnlq[:, :, 2, 1]
et_ps.L1.basis.embed.Rnl.connection.W[:, :, 4] = ps.rbasis.Wnlq[:, :, 2, 2]

et_ps.Ei.W[1, :, 1] .= ps.WB[:, 1]
et_ps.Ei.W[1, :, 2] .= ps.WB[:, 2]

##

calc_model = ACEpotentials.ACEPotential(model, ps, st)
rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)

##
# Test energy first

println("Testing energy...")

sys = rand_struct()
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

E_ref = ustrip(AtomsCalculators.potential_energy(sys, calc_model))
E_new = et_model(G, et_ps, et_st)[1]

println("E_ref = $E_ref")
println("E_new = $E_new")
println("E_err = $(abs(E_ref - E_new))")
@test abs(E_ref - E_new) < 1e-5

##
# Test forces and virial via Zygote

println("\nTesting forces via Zygote...")

# Define virial helper
function virial_from_edge_grads(G::ET.ETGraph, ∇E_edges)
   T = eltype(∇E_edges[1].𝐫)
   virial = @SMatrix zeros(T, 3, 3)
   for (edge_data, ∇E_edge) in zip(G.edge_data, ∇E_edges)
      𝐫ij = edge_data.𝐫
      ∂E_∂𝐫 = ∇E_edge.𝐫
      virial -= ∂E_∂𝐫 * 𝐫ij'
   end
   return virial
end

# Reference values
efv_ref = AtomsCalculators.energy_forces_virial(sys, calc_model)
F_ref = ustrip.(efv_ref.forces)
V_ref = ustrip(efv_ref.virial)

println("F_ref[1] = $(F_ref[1])")
println("V_ref = $V_ref")

# Compute gradient via Zygote
# The issue is that Zygote needs to differentiate through the edge_data
# which contains NamedTuples with position vectors.
# We need to make a simpler test first.

println("\nAttempting gradient via Zygote...")

# First let's try gradient w.r.t. parameters (should work)
println("Testing gradient w.r.t. parameters...")
try
   ∇ps = Zygote.gradient(ps -> et_model(G, ps, et_st)[1], et_ps)[1]
   println("Parameter gradient succeeded!")
   println("∇ps has keys: $(keys(∇ps))")
catch e
   println("Parameter gradient failed: $e")
end

# Now try gradient w.r.t. edge positions
# We need to create a wrapper that extracts just the position vectors
println("\nTesting gradient w.r.t. edge positions...")

# Extract position vectors from edge_data
𝐫_edges = [ed.𝐫 for ed in G.edge_data]
s0_edges = [ed.s0 for ed in G.edge_data]
s1_edges = [ed.s1 for ed in G.edge_data]

function _energy_from_positions(𝐫_vec)
   # Reconstruct edge_data from position vectors and species
   edge_data = [(𝐫 = r, s0 = s0, s1 = s1) for (r, s0, s1) in zip(𝐫_vec, s0_edges, s1_edges)]
   G_new = ET.ETGraph(G.ii, G.jj, G.first, G.node_data, edge_data, G.graph_data, G.maxneigs)
   return et_model(G_new, et_ps, et_st)[1]
end

try
   ∇𝐫 = Zygote.gradient(_energy_from_positions, 𝐫_edges)[1]

   if ∇𝐫 === nothing
      println("WARNING: Zygote returned nothing for position gradient")
   else
      println("Position gradient succeeded!")
      println("∇𝐫[1] = $(∇𝐫[1])")

      # Convert to forces using scatter
      # Given 𝐫_ij = X_j - X_i, and F = -∂E/∂X:
      # F[i] = -∂E/∂X_i = +∂E/∂𝐫_ij (since ∂𝐫_ij/∂X_i = -I)
      # F[j] = -∂E/∂X_j = -∂E/∂𝐫_ij (since ∂𝐫_ij/∂X_j = +I)
      T = eltype(∇𝐫[1])
      F_new = zeros(SVector{3, T}, length(sys))
      for (k, (i, j)) in enumerate(zip(G.ii, G.jj))
         F_new[i] += ∇𝐫[k]
         F_new[j] -= ∇𝐫[k]
      end
      println("F_new[1] = $(F_new[1])")

      # Convert to virial
      virial = @SMatrix zeros(T, 3, 3)
      for (k, (𝐫ij, ∂E_∂𝐫)) in enumerate(zip(𝐫_edges, ∇𝐫))
         virial -= ∂E_∂𝐫 * 𝐫ij'
      end
      println("V_new = $virial")

      # Check errors
      F_err = maximum(norm(f1 - f2) for (f1, f2) in zip(F_ref, F_new))
      V_err = maximum(abs.(V_ref - virial))

      println("\nF_err = $F_err")
      println("V_err = $V_err")

      @test F_err < 1e-5
      @test V_err < 1e-5
   end
catch e
   println("Position gradient failed: $e")
   showerror(stdout, e, catch_backtrace())
end

println("\nDone!")
