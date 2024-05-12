

using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

using Test, ACEbase
using ACEbase.Testing: print_tf, println_slim

using ACEpotentials
M = ACEpotentials.Models

using Optimisers, ForwardDiff

using Random, LuxCore, StaticArrays, LinearAlgebra
rng = Random.MersenneTwister(1234)

##

elements = (:Si, :O)
level = M.TotalDegree()
max_level = 15
order = 3 

model = M.ace_model(; elements = elements, order = order, Ytype = :solid, 
                      level = level, max_level = max_level, maxl = 8, 
                      init_WB = :glorot_normal)

ps, st = LuxCore.setup(rng, model)

calc = M.ACEPotential(model)

##

at = bulk(:Si, cubic=true) * 2
evf = M.energy_forces_virial(at, calc, ps, st)