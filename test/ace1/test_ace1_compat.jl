
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
# First multi-species examples 

params = ( elements = [:Al, :Ti,], 
           order = 3, 
           totaldegree = 8, 
           pure = false, 
           pure2b = false,
         )

ACE1_TestUtils.check_compat(params) 

## [5] 
# second multi-species example with three elements 
# and a few small changes to the basis 

params = ( elements = [:Al, :Ti, :C], 
           order = 2, 
           totaldegree = 6, 
           pure = false, 
           pure2b = false,
         )

ACE1_TestUtils.check_compat(params) 


##

using Random, Test, ACEbase, LinearAlgebra, Lux, Plots
using ACEbase.Testing: print_tf, println_slim
import ACE1, ACE1x, JuLIP

using ACEpotentials 
M = ACEpotentials.Models 
ACE1compat = ACEpotentials.ACE1compat
rng = Random.MersenneTwister(1234)

##

params = ( elements = [:Al, :Ti, :Cu], 
           order = 2, 
           totaldegree = 6, 
           pure = false, 
           pure2b = false,
         )

model1 = acemodel(; params...)
params2 = (; params..., totaldegree = params.totaldegree .+ 1)
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
# println_slim(@test basiserr < .3e-2)

##

rbasis1 = model1.basis.BB[2].pibasis.basis1p.J
rbasis2 = model2.rbasis
z11 = AtomicNumber(:Al)
z12 = Int(z1)
z21 = AtomicNumber(:Ti)
z22 = Int(z21)

rr = range(0.001, 5.0, length=200)
R1 = reduce(hcat, [ ACE1.evaluate(rbasis1, r, z11, z21) for r in rr])
R2 = reduce(hcat, [ rbasis2(r, z12, z22, NamedTuple(), NamedTuple()) for r in rr])

# alternating basis functions must be zero.
for n = 1:6
   @assert norm(R2[3*(n-1) + 1, :]) == 0
   @assert norm(R2[3*(n-1) + 3, :]) == 0
   @assert norm(R2[3*(n-1) + 2, :]*sqrt(2) - R1[n, :])/n^2  < 1e-3
end

for (i_nl, nl) in enumerate(rbasis2.spec)
   @assert R2[i_nl, :] ≈ R2[nl.n, :]
end 

# plt = plot() 
# for n = 1:6
#    plot!(R1[n, :], c = n, label = "R1,$n")
#    plot!(R2[3*(n-1)+2, :]*sqrt(2), c = n, ls = :dash, label = "R2,$n")
# end
# plt



