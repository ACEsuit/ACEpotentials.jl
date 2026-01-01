# using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "..", "EquivariantTensors.jl"))
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "..", "Polynomials4ML.jl"))
# Pkg.develop(url = joinpath(@__DIR__(), "..", "..", "DecoratedParticles"))

##

using ACEpotentials, StaticArrays, Lux, AtomsBase, AtomsBuilder, Unitful, 
      AtomsCalculators, Random, LuxCore, Test, LinearAlgebra, ACEbase, 
      ForwardDiff 

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

Random.seed!(1234)  # new seed to make sure the tests are consistent
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
# convert the pair basis to a splined version  

# overkill spline accuracy to check errors 
Nspl = 200

# polynomial basis taking y = y(r) as input 
polys_y = et_pair.rembed.layer.rbasis.basis
# weights for cat-1 
WW = et_ps.rembed.rbasis.post.W
splines = [ 
      P4ML.splinify( y -> WW[:, :, i] * polys_y(y), -1.0, 1.0, Nspl ) 
      for i in 1:size(WW, 3)  ]
states = [ P4ML._init_luxstate(spl) for spl in splines ]
selector2 = et_pair.rembed.layer.rbasis.post.selector
trans_y = et_pair.rembed.layer.rbasis.trans
envelope = et_pair.rembed.layer.envelope

spl_rbasis = ET.trans_splines(trans_y, splines, selector2, envelope)
ps_spl, st_spl = LuxCore.setup(rng, spl_rbasis)

poly_rbasis = et_pair.rembed.layer
ps_poly = et_ps.rembed
st_poly = et_st.rembed

## 

function rand_X() 
   sys = AtomsBuilder.bulk(:Si) * (2,2,1)
   rattle!(sys, 0.2u"Å") 
   AtomsBuilder.randz!(sys, [:Si => 0.5, :O => 0.5])
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   return G.edge_data 
end 


##

@info("Checking spline accuracy against polynomial basis")
Random.seed!(1234)  # new seed to make sure the tests are ok.
for ntest = 1:30 
   X = rand_X()
   P1, _ = poly_rbasis(X, ps_poly, st_poly)
   P2, _ = spl_rbasis(X, ps_spl, st_spl)
   spl_err = abs.(P1 - P2) ./ (abs.(P1) .+ abs.(P2) .+ 1)
   # @show maximum(spl_err)
   print_tf(@test maximum(spl_err) < 1e-5)

   (P1a, dP1), _ = ET.evaluate_ed(poly_rbasis, X, ps_poly, st_poly)
   (P2a, dP2), _ = ET.evaluate_ed(spl_rbasis, X, ps_spl, st_spl)
   print_tf(@test P2 ≈ P2a) 
   dspl_err = norm.(dP1 - dP2) ./ (1 .+ abs.(P1) + abs.(P2))
   # @show maximum(dspl_err)
   print_tf(@test maximum(dspl_err) < 1e-3)
end

##

@info("Checking machine precision derivative accuracy ")
# NOTE: This test should really be in ET and not here ... 

X = rand_X()
rand_u() = ( u = (@SVector randn(3)); DP.VState(𝐫 = u/norm(u)) )
U = [ rand_u() for _ = 1:length(X) ]

f(t) = spl_rbasis(X + t * U, ps_spl, st_spl)[1]
df0 = ForwardDiff.derivative(f, 0.0)

(P2a, dP2), _ = ET.evaluate_ed(spl_rbasis, X, ps_spl, st_spl)
dp = [ dot(U[i], dP2[i, j]) for i in 1:length(U), j = 1:size(dP2, 2) ]
println_slim(@test df0 ≈ dp) 


##
