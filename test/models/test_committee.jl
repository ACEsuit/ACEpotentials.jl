
using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ".."))
##

using Test, ACEbase, LinearAlgebra
using Polynomials4ML.Testing: print_tf, println_slim
using ACEpotentials
M = ACEpotentials.Models
using Optimisers, Unitful, AtomsCalculators, AtomsBase, AtomsBuilder
using AtomsCalculators: potential_energy, forces, virial 
using Statistics
AB = AtomsBuilder

# using Random, LuxCore, StaticArrays, LinearAlgebra
# rng = Random.MersenneTwister(1234)

##

elements = (:Si, :O)
max_level = 10
order = 3
E0s = Dict( :Si => -158.54496821u"eV", 
            :O => -2042.0330099956639u"eV")
NZ = length(elements)

model = ace1_model(; elements = elements, order = order, 
                     totaldegree = max_level, 
                     E0s = E0s,)

ps_vec, _restruct = destructure(model.ps)
ps_vec = randn(length(ps_vec)) ./ (1:length(ps_vec)).^2
co_ps_vec = [ ps_vec + 0.01 * randn(length(ps_vec)) ./ (1:length(ps_vec)).^2
              for _ = 1:10 ]
set_parameters!(model, ps_vec)
set_committee!(model, co_ps_vec)


sys = rattle!(bulk(:Si, cubic=true) * 2, 0.1) 
sys = randz!(sys, [:Si => 0.75, :O => 0.25])
potential_energy(sys, model)              
E, co_E = @committee potential_energy(sys, model)
@show (E - mean(co_E)) / length(sys)
@show cov(co_E/length(sys))
F, co_F = @committee forces(sys, model)
F̄ = mean(co_F)
@show maximum( norm.(F - F̄) )
@show cov( [ustrip.(f[1]) for f in co_F] )

