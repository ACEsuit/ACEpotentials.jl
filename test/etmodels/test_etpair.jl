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

# 1: polynomials without the envelope
# 
dp_agnesi = ETM._convert_agnesi(basis)
polys = basis.polys
selector2 = let zlist = et_zlist
   xij -> ET.catcat2idx(zlist, xij.z0, xij.z1)
end
et_linl = ET.SelectLinL(length(polys),         # indim
                        length(basis),       # outdim
                        NZ^2,                     # num (Zi,Zj) pairs
                        selector2)
rbasis_1 = ET.EmbedDP(dp_agnesi, polys, et_linl)

# 2: envelope 
_env_r = ETM._convert_envelope(basis.envelopes)
dp_envelope = ET.dp_transform( (x, st) -> _env_r.f( norm(x.𝐫), st ), _env_r.refstate )

# 3. combine into the radial basis 
et_rbasis = ETM.EnvRBranchL(dp_envelope, rbasis_1)

# convert this into an edge embedding                         
rembed = ET.EdgeEmbed( et_rbasis )
et_ps, et_st = Lux.setup(rng, rembed)

## 

function rand_struct() 
   sys = AtomsBuilder.bulk(:Si) * (2,2,1)
   rattle!(sys, 0.2u"Å") 
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys 
end 


sys = rand_struct()
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
out = rembed(G, et_ps, et_st)  # just a test run

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

et_pair = ETM.ETPairModel(rembed, readout)
et_ps, et_st = Lux.setup(rng, et_pair)

et_pair(G, et_ps, et_st)  # test run

##
# fixup the parameters to match the ACE model - here we incorporate the 
# 

# radial basis parameters for et_model_2 
et_ps.rembed.rbasis.post.W[:, :, 1] = ps.pairbasis.Wnlq[:, :, 1, 1]
et_ps.rembed.rbasis.post.W[:, :, 2] = ps.pairbasis.Wnlq[:, :, 1, 2]
et_ps.rembed.rbasis.post.W[:, :, 3] = ps.pairbasis.Wnlq[:, :, 2, 1]
et_ps.rembed.rbasis.post.W[:, :, 4] = ps.pairbasis.Wnlq[:, :, 2, 2]

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
   Ei, _ = et_model(G, et_ps, et_st)
   return sum(Ei) 
end

##

@info("Check total energies match")
for ntest = 1:30 
   sys = rand_struct()
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   E1 = AtomsCalculators.potential_energy(sys, calc_model)
   E2 = energy_new(sys, et_pair)
   print_tf( @test abs(ustrip(E1) - ustrip(E2)) < 1e-6 ) 
end

##

@info("Check gradients and jacobians")

sys = rand_struct()
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
nnodes = length(G.node_data)
iZ = et_pair.readout.selector.(G.node_data)
WW = et_ps.readout.W

# gradient of model w.r.t. positions 
∂G = ETM.site_grads(et_pair, G, et_ps, et_st)  # test run

# basis 
𝔹1 = ETM.site_basis(et_pair, G, et_ps, et_st)

# basis jacobian 
𝔹2, ∂𝔹2 = ETM.site_basis_jacobian(et_pair, G, et_ps, et_st)

println_slim(@test 𝔹1 ≈ 𝔹2)

∇Ei2 = reduce( hcat, ∂𝔹2[:, i, :] * WW[1, :, iZ[i]] 
                    for (i, iz) in enumerate(iZ) )
∇Ei3 = reshape(∇Ei2, size(∇Ei2)..., 1)
∇E_𝔹_edges = ET.rev_reshape_embedding(∇Ei3, G)[:]
println_slim(@test all(∇E_𝔹_edges .≈ ∂G.edge_data))

