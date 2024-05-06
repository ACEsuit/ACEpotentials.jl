
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

using ACEpotentials
M = ACEpotentials.Models

using Random, LuxCore
rng = Random.MersenneTwister(1234)

##

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 12
order = 3 

model = M.ace_model(; elements = elements, order = order, Ytype = :solid, 
                      level = level, max_level = max_level, maxl = 4)

ps, st = LuxCore.setup(rng, model)

# TODO: the number of parameters seems off. 

##

