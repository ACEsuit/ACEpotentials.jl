using ACEpotentials, Test
using Random: seed! 

@info(" ============== Testing for ACEpotentials 208 ================")
@info(" On Julia 1.9 some energy computations were inconsistent. ")   

model = acemodel(elements = [:Ti, ],
					  order = 3,
					  totaldegree = 10,
					  rcut = 6.0,
					  Eref = [:Ti => -1586.0195, ])

# generate random parameters 
seed!(1234)
params = randn(length(model.basis))
# params = params ./ (1:length(params)).^2    # (this is not needed)
ACEpotentials._set_params!(model, params)

function energy_per_at(pot, i) 
   at = bulk(:Ti) * i 
   return JuLIP.energy(pot, at) / length(at)
end

E_per_at = [ energy_per_at(model.potential, i) for i = 1:10 ]

maxdiff = maximum(abs(E_per_at[i] - E_per_at[j]) for i = 1:10, j = 1:10 )
@show maxdiff 

@test maxdiff < 1e-12

@info(" ============================================================")
