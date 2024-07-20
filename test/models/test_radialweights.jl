using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))

##

using ACEpotentials
M = ACEpotentials.Models

using Random, LuxCore, Test, ACEbase, LinearAlgebra
using ACEbase.Testing: print_tf
rng = Random.MersenneTwister(1234)

##

max_level = 8 
level = M.TotalDegree()
maxl = 3; maxn = max_level; 
elements = (:Si, )
order = 3 

model = M.ace_model(; elements = elements, order = order, Ytype = :solid, 
            level = level, max_level = max_level, maxl = 8, pair_maxn = 15, 
            init_WB = :zeros, 
            init_Wpair = :zeros, 
            init_Wradial = :linear)

ps, st = LuxCore.setup(rng, model)


##

