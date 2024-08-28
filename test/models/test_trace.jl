
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))

##

using Test, ACEbase
using ACEbase.Testing: print_tf, println_slim
using AtomsBase, AtomsBuilder, AtomsCalculators
AB = AtomsBuilder

using ACEpotentials
M = ACEpotentials.Models

using Optimisers, ForwardDiff

using Random, LuxCore, StaticArrays, LinearAlgebra
rng = Random.MersenneTwister(1234)
Random.seed!(11)

##

elements = (:C, :O, :H)
maxlevel = 4
order = 4 

##

model = M.trace_model(; elements=elements, 
                        order=order, 
                        max_level = maxlevel, 
                        maxn = 6, 
                        pair_maxn = 10, ) 

potential = M.ACEPotential(model)   

sys = AB.bulk(:C, cubic=true)*2
AB.randz!(sys, [:C => 0.4, :O => 0.2, :H => 0.4])
AB.rattle!(sys, 0.1)

potential.ps.WB[:,:] .= randn(size(potential.ps.WB))
efv = M.energy_forces_virial(sys, potential)