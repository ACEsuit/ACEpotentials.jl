using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
using TestEnv; TestEnv.activate();
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "..", "EquivariantTensors.jl"))
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "..", "Polynomials4ML.jl"))
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
# fixup the parameters to match the ACE model 

et_pair = ETM.convertpair(model)
et_ps, et_st = Lux.setup(rng, et_pair)

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
# make a splined version of the et_pair model 
# 

spl_50 = ETM.splinify(et_pair, et_ps, et_st; Nspl = 50)
ps_50, st_50 = Lux.setup(rng, spl_50)

spl_200 = ETM.splinify(et_pair, et_ps, et_st; Nspl = 200)
ps_200, st_200 = Lux.setup(rng, spl_200)

# many-body basis parameters for et_model_2
ps_50.readout.W[:] = et_ps.readout.W
ps_200.readout.W[:] = et_ps.readout.W


##
#
# test energy evaluations 
# 

calc_model = ACEpotentials.ACEPotential(model, ps, st)

function rand_struct() 
   sys = AtomsBuilder.bulk(:Si) * (2,2,1)
   rattle!(sys, 0.2u"Å") 
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys 
end 

function energy_new(sys, et_model, ps, st)
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   Ei, _ = et_model(G, ps, st)
   return sum(Ei) 
end

##

Random.seed!(1234)
@info("Check total energies match")
for ntest = 1:30 
   sys = rand_struct()
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   E1 = AtomsCalculators.potential_energy(sys, calc_model)
   E2 = energy_new(sys, et_pair, et_ps, et_st)
   E_50 = energy_new(sys, spl_50, ps_50, st_50)
   E_200 = energy_new(sys, spl_200, ps_200, st_200)
   print_tf( @test abs(ustrip(E1) - ustrip(E2)) < 1e-6 ) 
   print_tf( @test abs(ustrip(E2) - ustrip(E_50)) < 1e-2 )
   print_tf( @test abs(ustrip(E2) - ustrip(E_200)) < 1e-4 )
end

##

@info("Check gradients and jacobians")

sys = rand_struct()
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
nnodes = length(G.node_data)
iZ = et_pair.readout.selector.(G.node_data)
WW = et_ps.readout.W

# gradient of model w.r.t. positions 
∂G = ETM.site_grads(et_pair, G, et_ps, et_st) 
∂G_200 = ETM.site_grads(spl_200, G, ps_200, st_200)

# basis 
𝔹1 = ETM.site_basis(et_pair, G, et_ps, et_st)
𝔹1_200 = ETM.site_basis(spl_200, G, ps_200, st_200)

# basis jacobian 
𝔹2, ∂𝔹2 = ETM.site_basis_jacobian(et_pair, G, et_ps, et_st)
𝔹2_200, ∂𝔹2_200 = ETM.site_basis_jacobian(spl_200, G, ps_200, st_200)

println_slim(@test 𝔹1 ≈ 𝔹2)
println_slim(@test 𝔹1_200 ≈ 𝔹2_200)
println_slim(@test norm(𝔹1 - 𝔹1_200, Inf) < 1e-4)

∇Ei2 = reduce( hcat, ∂𝔹2[:, i, :] * WW[1, :, iZ[i]] 
                    for (i, iz) in enumerate(iZ) )
∇Ei3 = reshape(∇Ei2, size(∇Ei2)..., 1)
∇E_𝔹_edges = ET.rev_reshape_embedding(∇Ei3, G)[:]
println_slim(@test all(∇E_𝔹_edges .≈ ∂G.edge_data))

# check error in site energy gradients for splines 
println_slim(@test maximum(norm.(∂G.edge_data - ∂G_200.edge_data)) < 1e-3)
# check error in basis jacobian for splines 
println_slim(@test maximum(norm.(∂𝔹2 - ∂𝔹2_200)) < 1e-3)

##


# turn off during CI -- need to sort out CI for GPU tests 

#= 

@info("Check GPU evaluation") 
using Metal 
dev = Metal.mtl
ps_32 = ET.float32(et_ps)
st_32 = ET.float32(et_st)
ps_dev = dev(ps_32)
st_dev = dev(st_32)

sys = rand_struct()
G = ET.Atoms.interaction_graph(sys, 5.0 * u"Å")
G_32 = ET.float32(G)
G_dev = dev(G_32)

E1, st = et_pair(G_32, ps_32, st_32)
E2_dev, st_dev = et_pair(G_dev, ps_dev, st_dev)
E2 = Array(E2_dev)
println_slim(@test E1 ≈ E2)

##

@info(" .... with splines")
ps_50_32 = ET.float32(ps_50)
st_50_32 = ET.float32(st_50)
ps_50_dev = dev(ET.float32(ps_50))
st_50_dev = dev(ET.float32(st_50))
E3a, _ = spl_50(G_32, ps_50_32, st_50_32)
E3b_dev, _ = spl_50(G_dev, ps_50_dev, st_50_dev)
E3b = Array(E3b_dev)
println_slim(@test E3a ≈ E3b)

## 

@info(" .... gradients on GPU")
g1 = ETM.site_grads(et_pair, G_32, ps_32, st_32)
g2_dev = ETM.site_grads(et_pair, G_dev, ps_dev, st_dev)
g2_edge = Array(g2_dev.edge_data)
println_slim(@test all(g1.edge_data .≈ g2_edge))

@info(" .... basis on GPU")
b1 = ETM.site_basis(et_pair, G_32, ps_32, st_32)
b2_dev = ETM.site_basis(et_pair, G_dev, ps_dev, st_dev)
b2 = Array(b2_dev)
println_slim(@test b1 ≈ b2)

b1, ∂db1 = ETM.site_basis_jacobian(et_pair, G_32, ps_32, st_32)
b2_dev, ∂db2_dev = ETM.site_basis_jacobian(et_pair, G_dev, ps_dev, st_dev)
b2 = Array(b2_dev)
∂db2 = Array(∂db2_dev)
println_slim(@test b1 ≈ b2)
jacerr = norm.(∂db1 .- ∂db2) ./ (1 .+ norm.(∂db1) + norm.(∂db2))
@show maximum(jacerr)
println_slim( @test maximum(jacerr) < 1e-4 )

##
=#