
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
