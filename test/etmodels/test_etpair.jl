using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
using TestEnv; TestEnv.activate();
Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "..", "EquivariantTensors.jl"))
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "Polynomials4ML.jl"))
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "DecoratedParticles"))

##

using ACEpotentials, StaticArrays, Lux, AtomsBase, AtomsBuilder, Unitful, 
      AtomsCalculators, Random, LuxCore, Test, LinearAlgebra, ACEbase 

M = ACEpotentials.Models
ETM = ACEpotentials.ETModels
import EquivariantTensors as ET
import Polynomials4ML as P4ML 
import DecoratedParticles as DP

using Polynomials4ML.Testing: print_tf, println_slim

rng = Random.MersenneTwister(1234)
Random.seed!(1234)

##

# Generate an ACE model in the v0.8 style but 
#  - with fixed rcut. (relaxe this requirement later!!)
# get the pair potential component, compare with ETPairModel 
# make pair_learnable = true to prevent splinification. 

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 10
order = 3 
maxl = 6
rcut = 5.5

# modify rin0cuts to have same cutoff for all elements 
# TODO: there is currently a bug with variable cutoffs 
#       (?is there? The radials seem fine? check again)
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)


model = M.ace_model(; elements = elements, order = order, 
            Ytype = :solid, level = level, max_level = max_level, 
            maxl = maxl, pair_maxn = max_level,
            rin0cuts = rin0cuts,   
            init_WB = :glorot_normal, init_Wpair = :glorot_normal, 
            pair_learnable = true )

ps, st = Lux.setup(rng, model)

# confirm that the E0s are all zero 
@assert all( collect(values(model.Vref.E0)) .== 0 )

# set the many-body parameters to zero to isolate the pair potential 
ps.WB[:] .= 0 

## 
#
# construct an ETPairModel that is consistent with `model`

basis = model.pairbasis
et_zlist = ChemicalSpecies.(basis._i2z)
NZ = length(et_zlist)

# 1: extract r = |x.𝐫|
trans = ET.dp_transform( x -> norm(x.𝐫) )

# 2: radial basis r -> y -> P(y)   \
#                    -----> env(r) -> P(y) * env(r) 
#
# 2a : define the agnesi transform y = y(r) 
dp_agnesi = ETM._convert_agnesi(basis)
r_agnesi = ET.st_transform( (r, st) -> ET.eval_agnesi(r, st), 
                            dp_agnesi.refstate.params[1] )
# 2b : extract the radial basis
polys = basis.polys
# 2c : extract the envelopes
f_env = ETM._convert_envelope(basis.envelopes)

et_rbasis = BranchLayer(
                f_env, 
                Chain(; agnesi = r_agnesi, polys = polys); 
                fusion = WrappedFunction(eP -> eP[2] .* eP[1])
               )

# 3 : construct the SelLinL layer 
selector2 = let zlist = et_zlist
   xij -> ET.catcat2idx(zlist, xij.z0, xij.z1)
end
et_linl = ET.SelectLinL(length(polys),         # indim
                        length(basis),       # outdim
                        NZ^2,                     # num (Zi,Zj) pairs
                        selector2)

et_basis = ET.EdgeEmbed( ET.EmbedDP(trans, et_rbasis, et_linl) ) 
et_ps, st_st = Lux.setup(rng, et_basis)

## 

function rand_struct() 
   sys = AtomsBuilder.bulk(:Si) * (2,2,1)
   rattle!(sys, 0.2u"Å") 
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys 
end 


sys = rand_struct()
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
et_basis(G, et_ps, st_st)  # just a test run

##
#
# Complete the pair model 

selector1 = let zlist = et_zlist
   x -> ET.cat2idx(zlist, x.z)
end
readout = ET.SelectLinL(
               length(basis), 
               1,                  # output dim (only one site energy per atom)
               NZ,     # number of categories = num species 
               selector1)            

et_pair = ETM.ETPairModel(et_basis, readout)
et_ps, st_st = Lux.setup(rng, et_pair)

et_pair(G, et_ps, st_st)  # test run

##
# fixup the parameters to match the ACE model - here we incorporate the 
# 

# radial basis parameters for et_model_2 
et_ps.rembed.post.W[:, :, 1] = ps.pairbasis.Wnlq[:, :, 1, 1]
et_ps.rembed.post.W[:, :, 2] = ps.pairbasis.Wnlq[:, :, 1, 2]
et_ps.rembed.post.W[:, :, 3] = ps.pairbasis.Wnlq[:, :, 2, 1]
et_ps.rembed.post.W[:, :, 4] = ps.pairbasis.Wnlq[:, :, 2, 2]

# many-body basis parameters for et_model_2
et_ps.readout.W[1, :, 1] .= ps.Wpair[:, 1]
et_ps.readout.W[1, :, 2] .= ps.Wpair[:, 2]

##
#
# test energy evaluations 
# 

calc_model = ACEpotentials.ACEPotential(model, ps, st)

function energy_new(sys, et_model)
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   Ei, _ = et_model(G, et_ps, st_st)
   return sum(Ei) 
end

sys = rand_struct() 
E1 = AtomsCalculators.potential_energy(sys, calc_model) |> ustrip 
E2 = energy_new(sys, et_pair)

##
