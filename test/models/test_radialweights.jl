using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))

##

using ACEpotentials
M = ACEpotentials.Models

using Random, LuxCore, Test, LinearAlgebra
using Polynomials4ML.Testing: print_tf
rng = Random.MersenneTwister(1234)

##

max_level = 8 
level = M.TotalDegree()
maxl = 3; maxn = max_level; maxq_fact = 2;  
elements = (:Si, )
order = 3 

model = M.ace_model(; elements = elements, order = order, Ytype = :solid, 
            level = level, max_level = max_level, maxl = 8, pair_maxn = 15, 
            maxq_fact = maxq_fact, 
            init_WB = :zeros, 
            init_Wpair = :zeros, 
            init_Wradial = :onehot)

ps, st = LuxCore.setup(rng, model)

@show size(ps.rbasis.Wnlq)
display(ps.rbasis.Wnlq[:, :, 1, 1])

##

max_level = 8 
level = M.TotalDegree()
maxl = 3; maxn = max_level + 4;
elements = (:Si, :O)
order = 3 

model = M.ace_model(; elements = elements, order = order, Ytype = :solid, 
            level = level, max_level = max_level, maxl = 8, maxn = maxn, 
            pair_maxn = 15, 
            init_WB = :zeros, 
            init_Wpair = :zeros, 
            init_Wradial = :onehot)

ps, st = LuxCore.setup(rng, model)

display(ps.rbasis.Wnlq[:,:,1,1])
size(ps.rbasis.Wnlq)