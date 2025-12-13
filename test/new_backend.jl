using Pkg; Pkg.activate(joinpath(@__DIR__(), ".."))
using TestEnv; TestEnv.activate();
Pkg.develop(url = joinpath(@__DIR__(), ".."))
Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "EquivariantTensors.jl"))
Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "Polynomials4ML.jl"))

##

using ACEpotentials
M = ACEpotentials.Models

# build a pure Lux Rnl basis compatible with LearnableRnlrzz
import EquivariantTensors as ET
import Polynomials4ML as P4ML 

using StaticArrays, Lux
using AtomsBase, AtomsBuilder, Unitful, AtomsCalculators

using Random, LuxCore, Test, LinearAlgebra, ACEbase 
using Polynomials4ML.Testing: print_tf, println_slim
rng = Random.MersenneTwister(1234)

Random.seed!(1234)

##

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 10
order = 3 
maxl = 6

# modify rin0cuts to have same cutoff for all elements 
# TODO: there is currently a bug with variable cutoffs 
#       (?is there? The radials seem fine? check again)
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)


model = M.ace_model(; elements = elements, order = order, 
            Ytype = :solid, level = level, max_level = max_level, 
            maxl = maxl, pair_maxn = max_level,
            rin0cuts = rin0cuts,  
            init_WB = :glorot_normal, init_Wpair = :glorot_normal)

ps, st = Lux.setup(rng, model)          

# Missing issues: 
#    Vref = 0  =>  this will not be tested 
#    pair potential will also not be tested 

# kill the pair basis for now 
for s in model.pairbasis.splines
   s.itp.itp.coefs[:] *= 0 
end

## 
# build the Rnl basis 
# here we build it from the model.rbasis, so we can exactly match it 
# but in the final implementation we will have to create it directly 

rbasis = model.rbasis
et_i2z = AtomsBase.ChemicalSpecies.(rbasis._i2z)
et_rbasis = M._convert_Rnl_learnable(rbasis; zlist = et_i2z, 
                                        rfun = x -> norm(x.𝐫) )

# TODO: this is cheating, but this set can probably be generated quite 
#       easily as part of the construction of et_rbasis. 
et_rspec = rbasis.spec

## 
# build the ybasis 

et_ybasis = Chain( 𝐫ij = ET.NTtransform(x -> x.𝐫), 
                   Y = model.ybasis )
et_yspec = P4ML.natural_indices(et_ybasis.layers.Y)

# combining the Rnl and Ylm basis we can build an embedding layer 
et_embed = ET.EdgeEmbed( BranchLayer(; 
               Rnl = et_rbasis, 
               Ylm = et_ybasis ) )

## 
# now build the linear ACE layer 

# Convert AA_spec from (n,l,m) format to (n,l) format for mb_spec
AA_spec = model.tensor.meta["𝔸spec"] 
et_mb_spec = unique([[(n=b.n, l=b.l) for b in bb] for bb in AA_spec])

et_mb_basis = ET.sparse_equivariant_tensor(
      L = 0,  # Invariant (scalar) output only
      mb_spec = et_mb_spec,
      Rnl_spec = et_rspec,
      Ylm_spec = et_yspec,
      basis = real
   )

# et_acel = ET.SparseACElayer(et_mb_basis, (1,))

# ------------------------------------------------
# readout layer : need to select which linear output to 
#   use based on the center atom species

# CO: doing it this way is type unstable and causes problems in 
#     the GPU kernel generation. 
# __zi = let zlist = (_i2z = et_i2z, )
#    x -> M._z2i(zlist, x.s)
# end
#
# et_readout = ET.SelectLinL(
#                      et_mb_basis.lens[1],  # input dim
#                      1,                    # output dim
#                      length(et_i2z),       # num species
#                      __zi ) 


et_readout_2 = let zlist = et_i2z
      __zi = x -> ET.cat2idx(zlist, x.s)
      ET.SelectLinL(
               et_mb_basis.lens[1],  # input dim
               1,                    # output dim
               length(et_i2z),       # num species
               __zi ) 
end


# finally build the full model from the two layers 
#
# TODO: there is a huge problem here; the read-out layer needs to know 
#       about the center species; need to figure out how to pass that information 
#       through to the ace layer
#

__sz(::Any) = nothing
__sz(A::AbstractArray) = size(A) 
__sz(x::Tuple) = __sz.(x)
dbglayer(msg = ""; show=false) = WrappedFunction(x ->
         begin 
            println("$msg : ", typeof(x), ", ", __sz(x))
            if show; display(x); end 
            return x 
         end ) 

et_basis = Lux.Chain(;   
            embed = et_embed,    # embedding layer 
              ace = et_mb_basis,   # ACE layer -> basis
            unwrp = WrappedFunction(x -> x[1]),  # unwrap the tuple 
            )

et_model = Lux.Chain( 
      L1 = Lux.BranchLayer(;
         basis = et_basis,
         nodes = WrappedFunction(G -> G.node_data),   # pass node data through
         ),
      Ei = et_readout_2, 
      E = WrappedFunction(sum),         # sum up to get a total energy 
  )
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

##
# fixup all the parameters to make sure they match 
# the basis ordering appears to be identical, but it is not clear it really 
# is because meta["mb_spec"] only gives the original ordering before basis 
# construction ... 
nnll = M.get_nnll_spec(model.tensor)
et_nnll = et_mb_basis.meta["mb_spec"]
@show nnll == et_nnll 

# but this is also identical ... 
@show model.tensor.A2Bmaps[1] == et_mb_basis.A2Bmaps[1]

# radial basis parameters 
et_ps.L1.basis.embed.Rnl.connection.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]
et_ps.L1.basis.embed.Rnl.connection.W[:, :, 2] = ps.rbasis.Wnlq[:, :, 1, 2]
et_ps.L1.basis.embed.Rnl.connection.W[:, :, 3] = ps.rbasis.Wnlq[:, :, 2, 1]
et_ps.L1.basis.embed.Rnl.connection.W[:, :, 4] = ps.rbasis.Wnlq[:, :, 2, 2]

# many-body basis parameters; because the readout layer doesn't know about 
# species yet we take a single parameter set; this needs to be fixed asap. 
# ps.WB[:, 2] .= ps.WB[:, 1]

et_ps.Ei.W[1, :, 1] .= ps.WB[:, 1]
et_ps.Ei.W[1, :, 2] .= ps.WB[:, 2]

##

# wrap the old ACE model into a calculator 
calc_model = ACEpotentials.ACEPotential(model, ps, st)

# we will also need to get the cutoff radius which we didn't track 
# (Another TODO!!!)
rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)

function rand_struct() 
   sys = AtomsBuilder.bulk(:Si) * (2,1,1)
   rattle!(sys, 0.2u"Å") 
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys 
end 

function energy_new(sys, et_model)
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   return et_model(G, et_ps, et_st)[1]
end

##

for ntest = 1:30
   sys = rand_struct()
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   E1 = AtomsCalculators.potential_energy(sys, calc_model)
   E2 = energy_new(sys, et_model)
   print_tf( @test abs(ustrip(E1) - ustrip(E2)) < 1e-5 )
end
println()

##
# =========================================================================
#  FORCES AND VIRIAL EVALUATION VIA ZYGOTE
# =========================================================================
#
#  Key insight: We differentiate through the et_model using Zygote.
#  The gradient w.r.t. edge_data gives us ∂E/∂𝐫ij which we then:
#    - scatter-add to get forces: F_i = -∑_j ∂E/∂𝐫ij
#    - outer-product-sum for virial: σ = -∑_ij (∂E/∂𝐫ij) ⊗ 𝐫ij
# =========================================================================

using Zygote

# Define a function to compute energy from the graph representation
function energy_from_graph(G, model, ps, st)
   return model(G, ps, st)[1]
end

# Define virial_from_edge_grads (analogous to forces_from_edge_grads)
function virial_from_edge_grads(G::ET.ETGraph, ∇E_edges)
   # virial = -∑_ij (∂E/∂𝐫ij) ⊗ 𝐫ij
   # where ⊗ is outer product: (3,) ⊗ (3,) -> (3,3)
   T = eltype(∇E_edges[1].𝐫)
   virial = @SMatrix zeros(T, 3, 3)

   for (edge_data, ∇E_edge) in zip(G.edge_data, ∇E_edges)
      𝐫ij = edge_data.𝐫  # position vector for this edge
      ∂E_∂𝐫 = ∇E_edge.𝐫    # gradient of energy w.r.t. position
      virial -= ∂E_∂𝐫 * 𝐫ij'   # outer product and accumulate
   end

   return virial
end

##
# Test forces and virial on CPU via Zygote

println("Testing forces and virial via Zygote (CPU)...")

for ntest = 1:10
   sys = rand_struct()
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

   # Reference: use ACEPotential calculator
   efv_ref = AtomsCalculators.energy_forces_virial(sys, calc_model)
   E_ref = ustrip(efv_ref.energy)
   F_ref = ustrip.(efv_ref.forces)
   V_ref = ustrip(efv_ref.virial)

   # New backend via Zygote
   E_new = energy_from_graph(G, et_model, et_ps, et_st)

   # Get gradient w.r.t. edge_data via Zygote
   # We need to differentiate w.r.t. the graph's edge data
   # This requires making a version that extracts edge_data explicitly
   function _energy_from_edge_data(edge_data)
      G_new = ET.ETGraph(G.ii, G.jj, G.first, G.node_data, edge_data, G.maxneigs)
      return et_model(G_new, et_ps, et_st)[1]
   end

   ∇E_edges = Zygote.gradient(_energy_from_edge_data, G.edge_data)[1]

   # Convert edge gradients to forces
   F_new = ET.Atoms.forces_from_edge_grads(sys, G, ∇E_edges)

   # Convert edge gradients to virial
   V_new = virial_from_edge_grads(G, ∇E_edges)

   # Check energy
   E_err = abs(E_ref - E_new)

   # Check forces (compare magnitudes)
   F_err = maximum(norm(f1 - f2) for (f1, f2) in zip(F_ref, F_new))

   # Check virial
   V_err = maximum(abs.(V_ref - V_new))

   print_tf( @test E_err < 1e-5 )
   print_tf( @test F_err < 1e-5 )
   print_tf( @test V_err < 1e-5 )
end
println()

##
# =========================================================================
#  GPU EVALUATION (requires Metal, CUDA, or other GPU backend)
# =========================================================================

# Try to load a GPU backend
gpu_available = false
dev = nothing

try
   using Metal
   dev = Metal.mtl
   gpu_available = true
   println("Using Metal GPU backend")
catch
   try
      using CUDA
      dev = CUDA.cu
      gpu_available = true
      println("Using CUDA GPU backend")
   catch
      println("No GPU backend available, skipping GPU tests")
   end
end

if gpu_available
   println("\nTesting GPU energy evaluation...")

   sys = rand_struct()
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   G_32 = ET.ETGraph(G.ii, G.jj, G.first, ET.float32.(G.node_data), ET.float32.(G.edge_data), G.maxneigs)

   # move all data to the device
   G_32_dev = dev(G_32)
   ps_dev = dev(ET.float32(et_ps))
   st_dev = dev(ET.float32(et_st))

   E1 = AtomsCalculators.potential_energy(sys, calc_model)
   E2 = energy_new(sys, et_model)
   E3 = et_model(G_32_dev, ps_dev, st_dev)[1]

   println_slim( @test abs(ustrip(E1) - ustrip(E2)) < 1e-5 )
   println_slim( @test abs(ustrip(E1) - ustrip(E3)) / (abs(ustrip(E1)) + abs(ustrip(E3)) + 1e-7) < 1e-5 )

   ##
   # GPU Forces and Virial via Zygote
   println("\nTesting GPU forces and virial via Zygote...")

   # Helper to convert GPU edge gradients to forces on CPU
   function forces_from_gpu_grads(sys, G, ∇𝐫_gpu)
      # Collect gradients back to CPU
      ∇𝐫 = collect(∇𝐫_gpu)
      T = eltype(∇𝐫[1])
      F = zeros(SVector{3, T}, length(sys))
      for (k, (i, j)) in enumerate(zip(G.ii, G.jj))
         F[i] += ∇𝐫[k]
         F[j] -= ∇𝐫[k]
      end
      return F
   end

   # Helper to convert GPU edge gradients to virial on CPU
   function virial_from_gpu_grads(G, 𝐫_edges, ∇𝐫_gpu)
      ∇𝐫 = collect(∇𝐫_gpu)
      T = eltype(∇𝐫[1])
      virial = @SMatrix zeros(T, 3, 3)
      for (𝐫ij, ∂E_∂𝐫) in zip(𝐫_edges, ∇𝐫)
         virial -= ∂E_∂𝐫 * 𝐫ij'
      end
      return virial
   end

   for ntest = 1:5
      sys = rand_struct()
      G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

      # Reference from CPU
      efv_ref = AtomsCalculators.energy_forces_virial(sys, calc_model)
      E_ref = ustrip(efv_ref.energy)
      F_ref = ustrip.(efv_ref.forces)
      V_ref = ustrip(efv_ref.virial)

      # Convert to Float32 for GPU
      G_32 = ET.ETGraph(G.ii, G.jj, G.first,
                        ET.float32.(G.node_data),
                        ET.float32.(G.edge_data), G.maxneigs)

      # Extract position vectors before moving to GPU
      𝐫_edges_32 = [ed.𝐫 for ed in G_32.edge_data]
      s0_edges = [ed.s0 for ed in G_32.edge_data]
      s1_edges = [ed.s1 for ed in G_32.edge_data]

      # Move to GPU
      𝐫_gpu = dev(𝐫_edges_32)
      ps_32 = ET.float32(et_ps)
      ps_dev = dev(ps_32)
      st_dev = dev(ET.float32(et_st))

      # Energy function that takes GPU position array
      function _energy_gpu(𝐫_vec)
         # Reconstruct edge_data (this part stays on CPU for now)
         edge_data = [(𝐫 = r, s0 = s0, s1 = s1) for (r, s0, s1) in zip(collect(𝐫_vec), s0_edges, s1_edges)]
         G_new = ET.ETGraph(G.ii, G.jj, G.first, G_32.node_data, edge_data, G.maxneigs)
         G_dev = dev(G_new)
         return et_model(G_dev, ps_dev, st_dev)[1]
      end

      # Get GPU energy
      E_gpu = Float64(_energy_gpu(𝐫_edges_32))

      # Get gradient via Zygote (this should work through the GPU computation)
      ∇𝐫 = Zygote.gradient(_energy_gpu, 𝐫_edges_32)[1]

      if ∇𝐫 !== nothing
         # Convert to forces and virial
         F_gpu = forces_from_gpu_grads(sys, G, ∇𝐫)
         V_gpu = virial_from_gpu_grads(G, 𝐫_edges_32, ∇𝐫)

         # Check errors (use looser tolerance for Float32)
         E_err = abs(E_ref - E_gpu) / (abs(E_ref) + 1e-10)
         F_err = maximum(norm(f1 - Float64.(f2)) for (f1, f2) in zip(F_ref, F_gpu))
         V_err = maximum(abs.(V_ref - Float64.(V_gpu)))

         print_tf( @test E_err < 1e-4 )  # Float32 tolerance
         print_tf( @test F_err < 1e-3 )  # Float32 tolerance
         print_tf( @test V_err < 1e-3 )  # Float32 tolerance
      else
         println("WARNING: Zygote returned nothing for GPU gradient")
         @test false
      end
   end
   println()
end 
