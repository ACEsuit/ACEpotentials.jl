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
#  - with fixed rcut. (relaxes this requirement later!!)
#  - remove E0s
#  - remove pair potential 

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
            init_WB = :glorot_normal, init_Wpair = :glorot_normal)

ps, st = Lux.setup(rng, model)          

# wrap the old ACE model into a calculator 
calc_model = ACEpotentials.ACEPotential(model, ps, st)


# Missing issues: 
#    Vref = 0  =>  this will not be tested 
#    pair potential will also not be tested 

# kill the pair basis for now 
for s in model.pairbasis.splines
   s.itp.itp.coefs[:] *= 0 
end

## 
#
# Convert the v0.8 model to an ET backend based model based on the 
# implementation in ETM 
#
et_model = ETM.convert2et(model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et_model)

##
# fixup all the parameters to make sure they match 
# the basis ordering appears to be identical, but it is not clear it really 
# is because meta["mb_spec"] only gives the original ordering before basis 
# construction ... something to look into. 
nnll = M.get_nnll_spec(model.tensor)
et_nnll = et_model.basis.meta["mb_spec"]
@info("Check basis ordering")
println_slim(@test nnll == et_nnll) 

# but this is also identical ... 
@info("Check symmetrization operator")
@show ( model.tensor.A2Bmaps[1] == et_model.basis.A2Bmaps[1] )

# radial basis parameters for et_model 
et_ps.rembed.post.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]
et_ps.rembed.post.W[:, :, 2] = ps.rbasis.Wnlq[:, :, 1, 2]
et_ps.rembed.post.W[:, :, 3] = ps.rbasis.Wnlq[:, :, 2, 1]
et_ps.rembed.post.W[:, :, 4] = ps.rbasis.Wnlq[:, :, 2, 2]

# many-body basis parameters for et_model
et_ps.readout.W[1, :, 1] .= ps.WB[:, 1]
et_ps.readout.W[1, :, 2] .= ps.WB[:, 2]

##

# setup two splined ACE models 

spl_50 = ETM.splinify(et_model, et_ps, et_st; Nspl = 50)
ps_50, st_50 = Lux.setup(rng, spl_50)
ps_50.readout.W[:] .= et_ps.readout.W[:]

spl_200 = ETM.splinify(et_model, et_ps, et_st; Nspl = 200)
ps_200, st_200 = Lux.setup(rng, spl_200)
ps_200.readout.W[:] .= et_ps.readout.W[:]

##


function rand_struct() 
   sys = AtomsBuilder.bulk(:Si) * (2,2,1)
   rattle!(sys, 0.2u"Å") 
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   return sys 
end 

function energy_new(sys, _model, _ps, _st)
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   Ei, _ = _model(G, _ps, _st)
   return sum(Ei) 
end

##

Random.seed!(1234)
@info("Check total energies match")
for ntest = 1:30 
   sys = rand_struct()
   E1 = ustrip(AtomsCalculators.potential_energy(sys, calc_model))
   E2 = energy_new(sys, et_model, et_ps, et_st)
   E3 = energy_new(sys, spl_50, ps_50, st_50)
   E4 = energy_new(sys, spl_200, ps_200, st_200)
   print_tf( @test abs(E1 - E2) < 1e-6 ) 
   print_tf( @test abs(E2 - E3) / (1+abs(E2)+abs(E3)) < 1e-2 )
   print_tf( @test abs(E2 - E4) / (1+abs(E2)+abs(E4)) < 1e-4 )
end
println() 

##
#
# Zygote gradient 
#
using Zygote, ForwardDiff

sys = rand_struct()
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
∂G2a = Zygote.gradient(G -> sum(et_model(G, et_ps, et_st)[1]), G)[1]
∂G2b = ETM.site_grads(et_model, G, et_ps, et_st)

@info("confirm consistency of Zygote and site_grads")
println(@test all(∂G2a.edge_data .≈ ∂G2b.edge_data))

##
# test gradient against ForwardDiff 

function grad_fd(G, model, ps, st)
   function _replace_edges(X, Rmat)
      Rsvec = [ SVector{3}(Rmat[:, i]) for i in 1:size(Rmat, 2) ]
      new_edgedata = [ DP.PState(𝐫 = 𝐫, z0 = x.z0, z1 = x.z1, 𝐒 = x.𝐒) 
                      for (𝐫, x) in zip(Rsvec, G.edge_data) ]
      return ET.ETGraph( X.ii, X.jj, X.first, 
                  X.node_data, new_edgedata, X.graph_data, 
                  X.maxneigs )
   end 

   function _energy(Rmat)
      G_new = _replace_edges(G, Rmat)
      return sum(model(G_new, ps, st)[1])
   end
      
   Rsvec = [ x.𝐫 for x in G.edge_data ]
   Rmat = reinterpret(reshape, eltype(Rsvec[1]), Rsvec)
   ∇E_fd = ForwardDiff.gradient(_energy, Rmat)
   ∇E_svec = [ SVector{3}(∇E_fd[:, i]) for i in 1:size(∇E_fd, 2) ]
   ∇E_edges = [ DP.VState(; 𝐫 = 𝐫) for 𝐫 in ∇E_svec ]
   return ET.ETGraph( G.ii, G.jj, G.first, 
               G.node_data, ∇E_edges, G.graph_data, 
               G.maxneigs )
end 

@info("confirm consistency of gradients with ForwardDiff")

∇E_fd = grad_fd(G, et_model, et_ps, et_st)
println(@test all(∇E_fd.edge_data .≈ ∂G2b.edge_data))

##
#
# sys = rand_struct()
@info("Testing basis and jacobian")

G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
nnodes = length(G.node_data)
iZ = et_model.readout.selector.(G.node_data)
WW = et_ps.readout.W

𝔹1 = ETM.site_basis(et_model, G, et_ps, et_st)
𝔹2, ∂𝔹2 = ETM.site_basis_jacobian(et_model, G, et_ps, et_st)

##

@info("confirm correctness of site basis")

println_slim(@test 𝔹1 ≈ 𝔹2)
Ei_a = [ dot(𝔹2[i, :], WW[1, :, iZ[i]])    for (i, iz) in enumerate(iZ) ]
Ei_b = et_model(G, et_ps, et_st)[1][:]
println_slim(@test Ei_a ≈ Ei_b)

##

@info("Confirm correctness of Jacobian against gradient")
# compute the gradient from the jacobian by hand 
#    size(𝔹2) = (num_nodes, basis_len)
#    size(∂𝔹2) = (num_edges, num_nodes, basislen)

∇Ei2 = reduce( hcat, ∂𝔹2[:, i, :] * WW[1, :, iZ[i]] 
                    for (i, iz) in enumerate(iZ) )
∇Ei3 = reshape(∇Ei2, size(∇Ei2)..., 1)
∇E_𝔹_edges = ET.rev_reshape_embedding(∇Ei3, G)[:]
println(@test all(∇E_𝔹_edges .≈ ∂G2b.edge_data))


## 
#
# demo GPU evaluation 
#

#
# turning off this test until we figure out how to do proper CI on GPUs?
# until then this just needs to be done manually and locally?

#=

@info("Checking GPU evaluation with Metal.jl")

# TODO: replace Metal with generic GPU test 
using Metal
dev = Metal.mtl

sys = rand_struct()
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
G_32 = ET.float32(G)

# move all data to the device 
G_32_dev = dev(G_32)
ps_dev = dev(ET.float32(et_ps))
st_dev = dev(ET.float32(et_st))
ps_32 = ET.float32(et_ps)
st_32 = ET.float32(et_st)

E1 = ustrip(AtomsCalculators.potential_energy(sys, calc_model))
E4 = sum(et_model(G_32_dev, ps_dev, st_dev)[1])
println_slim( @test abs(E1 - E4) / (abs(E1) + abs(E4) + 1e-7) < 1e-5 ) 

## 
# gradients on GPU 

@info("Check Evaluation of gradient on GPU")
g1 = ETM.site_grads(et_model, G_32, ps_32, st_32)
g2_dev = ETM.site_grads(et_model, G_32_dev, ps_dev, st_dev)
∇1 = g1.edge_data
∇2 = Array(g2_dev.edge_data)
println_slim( @test all(∇1 .≈ ∇2) )

## 

@info("Basis evaluation on GPU")

𝔹1 = ETM.site_basis(et_model, G_32, ps_32, st_32)
𝔹2_dev = ETM.site_basis(et_model, G_32_dev, ps_dev, st_dev)
𝔹2 = Array(𝔹2_dev)
println_slim( @test 𝔹1 ≈ 𝔹2 )

@info("Basis jacobian evaluation on GPU")
𝔹1, ∂𝔹1 = ETM.site_basis_jacobian(et_model, G_32, ps_32, st_32)
𝔹2_dev, ∂𝔹2_dev = ETM.site_basis_jacobian(et_model, G_32_dev, ps_dev, st_dev)

𝔹2 = Array(𝔹2_dev)
∂𝔹2 = Array(∂𝔹2_dev)

println_slim( @test 𝔹1 ≈ 𝔹2 )
err_jac = norm.(∂𝔹1 - ∂𝔹2) ./ (norm.(∂𝔹1) + norm.(∂𝔹2) .+ 0.1) 
println_slim( @test maximum(err_jac) < 1e-4 )
@show maximum(err_jac)
@info("The jacobian error feels a bit large. This may need further investigation.")

=#