using Pkg; Pkg.activate(joinpath(@__DIR__(), ".."))
using TestEnv; TestEnv.activate();
Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "EquivariantTensors.jl"))
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "Polynomials4ML.jl"))

##

using ACEpotentials
M = ACEpotentials.Models

# build a pure Lux Rnl basis compatible with LearnableRnlrzz
import EquivariantTensors as ET
import Polynomials4ML as P4ML 
import DecoratedParticles as DP

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
# et_rbasis = M._convert_Rnl_learnable(rbasis; zlist = et_i2z, 
#                                         rfun = x -> norm(x.𝐫) )
et_rbasis = M._convert_Rnl_learnable(rbasis)

# TODO: this is cheating, but this set can probably be generated quite 
#       easily as part of the construction of et_rbasis. 
et_rspec = rbasis.spec

##
# test a new implementation of the Rnl basis that is _ed differentiable 
# which is needed for jacobians 

psr, str = Lux.setup(rng, et_rbasis)

transr = et_rbasis.layers.layers.y
sellinl = et_rbasis.connection
polys = et_rbasis.layers.layers.Pe

et_rbasis3 = ET.EmbedDP(transr, P4ML.wrapped_basis(polys, rand()), sellinl)
psr3, str3 = Lux.setup(rng, et_rbasis3)
psr3.post.W[:] .= psr.connection.W[:]


X = [ DP.PState( 𝐫 = 2*randn(SVector{3, Float64}), z0 = rand(et_i2z), z1 = rand(et_i2z)) 
        for _ = 1:10 ]

R1, _ = et_rbasis(X, psr, str)

y = transr.(X, Ref(transr.refstate))
P, _ = polys(y, psr3.basis, str3.basis)
R2, _ = sellinl((P, X), psr3.post, str3.post)

R3, _ = et_rbasis3(X, psr3, str3)

@show R1 ≈ R2 ≈ R3 

## 

(R3a, ∂R3), _ = ET.evaluate_ed(et_rbasis3, X, psr3, str3)
R3a ≈ R3


## 
# build the ybasis 

et_ybasis = Chain( 𝐫ij = ET.NTtransform(x -> x.𝐫), 
                   Y = model.ybasis )
et_yspec = P4ML.natural_indices(et_ybasis.layers.Y)

##
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
#
# demo GPU evaluation 
#

using Metal
dev = Metal.mtl

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
