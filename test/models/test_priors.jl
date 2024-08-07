
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

##

using Test, ACEbase, Random 
using ACEbase.Testing: print_tf, println_slim
using Lux, LuxCore, StaticArrays, LinearAlgebra
rng = Random.MersenneTwister(1234)
Random.seed!(11)

using ACEpotentials
M = ACEpotentials.Models

##

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 10
order = 3 

model = M.ace_model(; elements = elements, order = order, Ytype = :spherical, 
            level = level, max_level = max_level, pair_maxn = max_level, init_WB = :glorot_normal, init_Wpair = :glorot_normal)
ps, st = LuxCore.setup(rng, model)

##

Γa = M.algebraic_smoothness_prior(model)
Γe = M.exp_smoothness_prior(model)
Γg = M.gaussian_smoothness_prior(model)

[Γa.diag Γe.diag Γg.diag]