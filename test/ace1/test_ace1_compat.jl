
using Pkg; Pkg.activate(joinpath(@__DIR__(), ".."))
# using TestEnv; TestEnv.activate();

##

using Plots
using Random, Test, ACEbase, LinearAlgebra, Lux
using ACEbase.Testing: print_tf, println_slim

using ACEpotentials
M = ACEpotentials.Models
ACE1compat = ACEpotentials.ACE1compat
rng = Random.MersenneTwister(1234)

##

params = ( elements = [:Si,], 
           order = 3, 
           transform = (:agnesi, 2, 2),
           totaldegree = [12, 10, 8], 
           pure = false, 
           pure2b = false,
           pair_envelope = (:r, 1),
           rcut = 5.5,
           Eref = [:Si => -1.234 ]
         )

##

ACE1_TestUtils.check_compat(params) 
