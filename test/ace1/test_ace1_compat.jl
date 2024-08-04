
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
# using TestEnv; TestEnv.activate();

##

include(@__DIR__() * "/ace1_testutils.jl")

@info(
""" 
==================================
=== Testing ACE1 compatibility === 
==================================
""")


##
# [1] 
# a first test that was used to write the original 
# ACE1 compat module and tests 

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

ACE1_TestUtils.check_compat(params) 

##
# [2] 
# same as [1] but with auto cutoff, a different pair envelope, 
# different transforms, and a different chemical element, 
# different choice of degrees, different element   
# 

params = ( elements = [:C,], 
           order = 3, 
           transform = (:agnesi, 2, 4),
           totaldegree = [14, 12, 10], 
           pure = false, 
           pure2b = false,
           pair_transform = (:agnesi, 1, 3), 
           pair_envelope = (:r, 3),
           Eref = [:C => -1.234 ]
         )

ACE1_TestUtils.check_compat(params) 

##
# [3] 
# A minimal example with as many defaults as possible
# 

params = ( elements = [:W,], 
           order = 2, 
           totaldegree = 10, 
           pure = false, 
           pure2b = false,
         )

ACE1_TestUtils.check_compat(params) 

## 
# [4] 
# A first multi-species example 

params = ( elements = [:Al, :Ti,], 
           order = 3, 
           totaldegree = 8, 
           pure = false, 
           pure2b = false,
         )

ACE1_TestUtils.check_compat(params) 

##

using Random, Test, ACEbase, LinearAlgebra, Lux
using ACEbase.Testing: print_tf, println_slim
import ACE1, ACE1x, JuLIP

using ACEpotentials 
M = ACEpotentials.Models 
ACE1compat = ACEpotentials.ACE1compat
rng = Random.MersenneTwister(1234)

##

params = ( elements = [:Al, :Ti,], 
           order = 3, 
           totaldegree = 7, 
           pure = false, 
           pure2b = false,
         )

model1 = acemodel(; params...)
params2 = (; params..., totaldegree = params.totaldegree .+ 0.1)
model2 = ACE1compat.ace1_model(; params...)
ps, st = Lux.setup(rng, model2)

lenB1 = length(model1.basis)
Nenv = 10 * lenB1

Random.seed!(12345)
XX2 = [ M.rand_atenv(model2, rand(6:10)) for _=1:Nenv ]
XX1 = [ (x[1], AtomicNumber.(x[2]), AtomicNumber(x[3])) for x in XX2 ]

B1 = reduce(hcat, [ ACE1_TestUtils._evaluate(model1.basis, x...) for x in XX1])
B2 = reduce(hcat, [ M.evaluate_basis(model2, x..., ps, st) for x in XX2])

@info("Compute linear transform between bases to show match") 
# We want full-rank C such that C * B2 = B1 
# (note this allows B2 > B1)
C = B2' \ B1'
basiserr = norm(B1 - C' * B2, Inf)
@show basiserr
println_slim(@test basiserr < .3e-2)


# ACE1_TestUtils.check_basis(model1, model2)

