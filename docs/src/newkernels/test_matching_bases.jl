
include("match_bases.jl")
using LinearAlgebra, StaticArrays, Lux

_evaluate(basis::JuLIP.MLIPs.IPSuperBasis, Rs, Zs, z0) = 
      reduce(vcat, [ACE1.evaluate(B, Rs, Zs, z0) for B in basis.BB])

##

Z0 = :Si 
z1 = AtomicNumber(Z0)
z2 = Int(z1)

ps, st = Lux.setup(Random.GLOBAL_RNG, model2)
model1, model2, calc_model2 = matching_bases(; Z = Z0)

##
# confirm match on atomic environments 

Nenv = 1000
XX2 = [ M.rand_atenv(model2, rand(6:10)) for _=1:Nenv ]
XX1 = [ (x[1], AtomicNumber.(x[2]), AtomicNumber(x[3])) for x in XX2 ]


B1 = reduce(hcat, [ _evaluate(model1.basis, x...) for x in XX1])
B2 = reduce(hcat, [M.evaluate_basis(model2, x..., ps, st) for x in XX2])

@info("Compute linear transform between bases to show match")
C = B1' \ B2'
@show norm(B2 - C' * B1)

@info("Transform should be a permuted diagonal, but this is not quite true...")
@show size(C)
@show count(abs.(C) .> 1e-10)

##

@info("Check match on a dataset (Zuo)")
# we will try this for a simple dataset, Zuo et al 
# replace element with any of those available in that dataset 

train, test, _ = ACEpotentials.example_dataset("Zuo20_$Z0")
train = train[1:10]

EB1 = reduce(hcat, [ ACE1.energy(model1.basis, sys) for sys in train ])
EB2 = reduce(hcat, [ M.energy_forces_virial_basis(FlexibleSystem(sys), calc_model2, ps, st).energy for sys in train ])

@show norm(ustrip.(EB2) - C' * EB1)

