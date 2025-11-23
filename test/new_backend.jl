using Pkg; Pkg.activate(joinpath(@__DIR__(), ".."))
using TestEnv; TestEnv.activate();

using ACEpotentials
M = ACEpotentials.Models

# build a pure Lux Rnl basis compatible with LearnableRnlrzz
import EquivariantTensors as ET
import Polynomials4ML as P4ML 
using StaticArrays, AtomsBase
using Lux

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

model = M.ace_model(; elements = elements, order = order, 
            Ytype = :solid, level = level, max_level = max_level, 
            maxl = maxl, pair_maxn = max_level, 
            init_WB = :glorot_normal, init_Wpair = :glorot_normal)

ps, st = Lux.setup(rng, model)          

# Missing issues: 
#    Vref = 0  =>  this will not be tested 

# kill the pair basis for now 
for s in model.pairbasis.splines
   s.itp.itp.coefs[:] *= 0 
end

## 
# build the Rnl basis 
# here we build it from the model.rbasis, so we can exactly match it 
# but in the final implementation we will have to create it directly 

rbasis = model.rbasis

# ET uses AtomsBase.ChemicalSpecies 
et_i2z = AtomsBase.ChemicalSpecies.(rbasis._i2z)

# In ET we currently store an edge xij as a NamedTuple, e.g, 
#    xij = (𝐫ij = ..., zi = ..., zj = ...)
# The NTtransform is a wrapper for mapping xij -> y 
# (in this case y = transformed distance) adding logic to enable 
# differentiation through this operation. 
#
# In ET.Atoms edges are of the form xij = (𝐫 = ..., s0 = ..., s1 = ...)
#
et_trans = let _i2z = (_i2z = et_i2z,), transforms = rbasis.transforms
   ET.NTtransform(x -> begin
         idx_i = M._z2i(_i2z, x.s0)
         idx_j = M._z2i(_i2z, x.s1)
         trans_ij = rbasis.transforms[idx_i, idx_j] 
         r = norm(x.𝐫)
         return trans_ij(r)
      end)
   end 

# the envelope is always a simple quartic (1 -x^2)^2
#  (note the transforms is normalized to map to [-1, 1])
et_env = y -> (1 - y^2)^2

# the polynomial basis 
et_polys = rbasis.polys

# the linear layer transformation 
# selector maps a (Zi, Zj) pair to an index a for transforming 
#   P(yij) -> W[a] * P(zij) 
# with W[a] learnable weights 
selector = let _i2z = (_i2z = et_i2z,)
   x -> begin 
         iz = M._z2i(_i2z, x.s0)
         jz = M._z2i(_i2z, x.s1)
         return (iz - 1) * length(_i2z) + jz
      end 
   end 
#                                          
et_linl = ET.SelectLinL(length(et_polys),         # indim
                        size(ps.rbasis.Wnlq, 1),  # outdim
                        4,                        # 4 categories
                        selector)

et_rbasis = SkipConnection(   # input is xij 
         Chain(y = et_trans,  # transforms yij 
               P = SkipConnection(
                     et_polys, 
                     WrappedFunction( Py -> et_env.(Py[2]) .* Py[1] )
                  )
               ),   # r -> y -> P = e(y) * polys(y)
         et_linl    # P -> W(Zi, Zj) * P 
      )

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

et_acel = ET.SparseACElayer(et_mb_basis, (1,))

# finally build the full model from the two layers 
#
# TODO: there is a huge problem here; the read-out layer needs to know 
#       about the center species; need to figure out how to pass that information 
#       through to the ace layer
#
et_model = Lux.Chain(; 
            embed = et_embed,  # embedding layer 
            ace = et_acel,     # ACE layer / correlation layer 
            energy = WrappedFunction(x -> sum(x[1]))   # sum up to get a total energy 
            )
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

##
# fixup all the parameters to make sure they match 
# the basis ordering appears to be identical, but it is not clear it really 
# is because meta["mb_spec"] only gives the original ordering before basis 
# construction ... 
nnll = M.get_nnll_spec(model.tensor)
et_nnll = et_model.layers.ace.symbasis.meta["mb_spec"]
@show nnll == et_nnll 

# radial basis parameters 
et_ps.embed.Rnl.connection.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]
et_ps.embed.Rnl.connection.W[:, :, 2] = ps.rbasis.Wnlq[:, :, 1, 2]
et_ps.embed.Rnl.connection.W[:, :, 3] = ps.rbasis.Wnlq[:, :, 2, 1]
et_ps.embed.Rnl.connection.W[:, :, 4] = ps.rbasis.Wnlq[:, :, 2, 2]

# many-body basis parameters; because the readout layer doesn't know about 
# species yet we take a single parameter set; this needs to be fixed asap. 
ps.WB[:, 2] .= ps.WB[:, 1]
et_ps.ace.WLL[1][:] .= ps.WB[:, 1]

##

# generate random structures 
using AtomsBuilder, Unitful, AtomsCalculators

# wrap the old ACE model into a calculator 
calc_model = ACEpotentials.ACEPotential(model, ps, st)

# we will also need to get the cutoff radius which we didn't track 
# (Another TODO!!!)
rcut = maximum(a.rcut for a in model.pairbasis.rin0cuts)

function rand_struct() 
   sys = AtomsBuilder.bulk(:Si, cubic=true) * 2
   rattle!(sys, 0.2u"Å") 
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys 
end 

function energy_new(sys, et_model)
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   return et_model(G, et_ps, et_st)[1]
end

sys = rand_struct()
AtomsCalculators.potential_energy(sys, calc_model)
energy_new(sys, et_model)

