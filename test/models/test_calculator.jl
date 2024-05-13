

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


##
# testing the AD through a loss function 


at = rattle!(bulk(:Si, cubic=true), 0.1)

using Unitful 
using Unitful: ustrip

wE = 1.0 / u"eV"
wV = 1.0 / u"eV"
wF = 0.33 / u"eV/Å"

function loss(at, calc, ps, st)
   efv = M.energy_forces_virial(at, calc, ps, st)
   _norm_sq(f) = sum(abs2, f)
   return (   wE^2 * efv.energy^2 / length(at) 
            + wV^2 * sum(abs2, efv.virial) / length(at)  
            + wF^2 * sum(_norm_sq, efv.forces) )
end

##

using Zygote 
Zygote.refresh() 

Zygote.gradient(ps -> loss(at, calc, ps, st), ps)[1] 
