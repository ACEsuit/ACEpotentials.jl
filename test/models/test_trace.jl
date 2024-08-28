

using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))

##

using Test, ACEbase
using ACEbase.Testing: print_tf, println_slim

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

NZ = length(elements)
rin0cuts = M._default_rin0cuts(elements)
maxn = 6
maxq = maxlevel * 2
level = M.TraceLevel(maxn)

rbasis = M.ace_learnable_Rnlrzz(; max_level = max_level, level = level, 
                                   maxl = maxlevel, maxn = maxn, 
                                   maxq = maxq, 
                                   elements = elements, 
                                   rin0cuts = rin0cuts, 
                                   Winit = :glorot_normal)


rspec = rbasis.spec  
aa_spec = M.sym_trace_spec(; order=order, r_spec = rspec, max_level = maxlevel)                                 

##

model = M.trace_model(; elements=elements, 
                        order=order, 
                        max_level = maxlevel, 
                        maxn = 6, 
                        pair_maxn = 10, ) 

