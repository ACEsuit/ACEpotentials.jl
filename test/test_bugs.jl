
using ACEpotentials, Test
using Random: seed! 
using ACEpotentials.ACE1compat: ace1_model
using ACEpotentials.Models: ACEPotential, potential_energy
using AtomsBuilder
using Unitful

@info(" ============== Testing for ACEpotentials #208 ================")
@info(" On Julia 1.9 some energy computations were inconsistent. ")   

model = ace1_model(elements = [:Ti, ],
					    order = 3,
					    totaldegree = 10,
					    rcut = 6.0,
					    Eref = [:Ti => -1586.0195, ])


# generate random parameters 
seed!(1234)
params = randn(ACEpotentials.length_basis(model))
# params = params ./ (1:length(params)).^2    # (this is not needed)
ACEpotentials.Models.set_parameters!(model, params)

function energy_per_at(pot, i) 
   at = bulk(:Ti) * i 
   return potential_energy(at, pot) / length(at)
end

E_per_at = [ energy_per_at(model, i) for i = 1:10 ]

maxdiff = maximum(abs(E_per_at[i] - E_per_at[j]) for i = 1:10, j = 1:10 )
@show maxdiff 

@test ustrip(u"eV", maxdiff) < 1e-9

@info(" ============================================================")
@info(" ============== Testing for no Eref bug ====================")

# there was never an issue filed for this, but it was an annoying issue 
# that came up twice by making changes in the model construction heuristics

params1 = (elements = [:Si], rcut = 5.5, order = 3, totaldegree = 12)
params2 = (; :Eref => [:Si => 0.0], pairs(params1)...)
model1 = ace1_model(; params1...) 
model2 = ace1_model(; params2...) 
sys = bulk(:Si, cubic=true) * 2
println_slim(@test potential_energy(sys, model1) == potential_energy(sys, model2) == 0.0u"eV")

@info(" ============================================================")
