# This script is to roughly document how to use the new model implementations 
# I'll try to explain what can be done and what is missing along the way. 
# I am 

using Random
using ACEpotentials, AtomsBase, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Zygote, Optimisers, Folds, Printf 

# we will try this for a simple dataset, Zuo et al 
# replace element with any of those available in that dataset 

Z0 = :Si 
train, test, _ = ACEpotentials.example_dataset("Zuo20_$Z0")
train = train[1:3:end]

# because the new implementation is experimental, it is not exported, 
# so I create a little shortcut to have easy access. 

M = ACEpotentials.Models

# First we create an ACE1 style potential with some standard parameters 

elements = [Z0,]
order = 3 
totaldegree = 10
rcut = 5.5 

model1 = acemodel(elements = elements, 
                  order = order, 
                  totaldegree = totaldegree, 
                  pure = false, pure2b = false, 
                  rcut = rcut,  )

# now we create an ACE2 style model that should behave similarly                   

# this essentially reproduces the rcut = 5.5, we may want a nicer way to 
# achieve this. 

rin0cuts = M._default_rin0cuts(elements; rcutfactor = 2.3)

model2 = M.ace_model(; elements = elements, 
                       order = order,               # correlation order 
                       Ytype = :solid,              # solid vs spherical harmonics
                       level = M.TotalDegree(),     # how to calculate the weights to give to a basis function
                       max_level = totaldegree,     # maximum level of the basis functions
                       pair_maxn = totaldegree,     # maximum number of basis functions for the pair potential 
                       init_WB = :zeros,            # how to initialize the ACE basis parmeters
                       init_Wpair = :zeros,         # how to initialize the pair potential parameters
                       init_Wradial = :linear, 
                       rin0cuts = rin0cuts, 
                     )

# wrap the model into a calculator, which turns it into a potential...

calc_model2 = M.ACEPotential(model2)

# Fit the ACE1 model 

# set weights for energy, forces virials 
weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),);
# specify a solver 
solver=ACEfit.TruncatedSVD(; rtol = 1e-8)

acefit!(model1, train;  solver=solver)



# Fit the ACE2 model - this still needs a bit of hacking to convert everything 
# to the new framework. 
# - convert the data to AtomsBase 
# - use a different interface to specify data weights and keys 
#   (this needs to be brough in line with the ACEpotentials framework)
# - rewrite the assembly for the LSQ system from scratch (but this is easy)

train2 = FlexibleSystem.(train)
test2 = FlexibleSystem.(test)
data_keys = (E_key = :energy, F_key = :force, ) 
weights = (wE = 30.0/u"eV", wF = 1.0 / u"eV/Å", )

function local_lsqsys(calc, at, ps, st, weights, keys)
   efv = M.energy_forces_virial_basis(at, calc, ps, st)

   # There are no E0s in this dataset! 
   # # compute the E0s contribution. This needs to be done more 
   # # elegantly and a stacked model would solve this problem. 
   # E0 = sum( calc.model.E0s[M._z2i(calc.model, z)] 
   #           for z in AtomsBase.atomic_number(at) ) * u"eV"

   # energy 
   wE = weights[:wE]
   E_dft = at.data[data_keys.E_key] * u"eV"
   y_E = wE * E_dft # (E_dft - E0)
   A_E = wE * efv.energy'

   # forces 
   wF = weights[:wF]
   F_dft = at.data[data_keys.F_key] * u"eV/Å"
   y_F = wF * reinterpret(eltype(F_dft[1]), F_dft)
   A_F = wF * reinterpret(eltype(efv.forces[1]), efv.forces)

   # # virial 
   # wV = weights[:wV]
   # V_dft = at.data[data_keys.V_key] * u"eV"
   # y_V = wV * V_dft[:]
   # # display( reinterpret(eltype(efv.virial), efv.virial) )
   # A_V = wV * reshape(reinterpret(eltype(efv.virial[1]), efv.virial), 9, :)

   return vcat(A_E, A_F), vcat(y_E, y_F)
end


function assemble_lsq(calc, data, weights, data_keys; 
                      rng = Random.MersenneTwister(1234), 
                      executor = Folds.ThreadedEx())
   ps, st = Lux.setup(rng, calc)
   blocks = Folds.map(at -> local_lsqsys(calc, at, ps, st, 
                                         weights, data_keys), 
                      data, executor)
   A = reduce(vcat, [b[1] for b in blocks])
   y = reduce(vcat, [b[2] for b in blocks])
   return A, y
end


A, y = assemble_lsq(calc_model2, train2[1:10], weights, data_keys)

θ = ACEfit.trunc_svd(svd(A), y, 1e-8)
ps, st = Lux.setup(rng, calc_model2)

# the next step is a hack. This should be automatable, probably using Lux.freeze. 
# But I couldn't quite figure out how to use that. 
# Here I'm manually constructing a parameters NamedTuple with rbasis removed. 
# then I'm using the destructure / restructure method from Optimizers to 
# convert θ into a namedtuple. 

ps_lin = (WB = ps.WB, Wpair = ps.Wpair, pairbasis = ps.pairbasis, rbasis = NamedTuple())
_, restruct = destructure(ps_lin) 
ps_lin_fit = restruct(θ)
ps_fit = deepcopy(ps)
ps_fit.WB[:] = ps_lin_fit.WB[:]
ps_fit.Wpair[:] = ps_lin_fit.Wpair[:]
calc_model2_fit = M.ACEPotential(model2, ps_fit, st)


# Now we can compare errors? 
# to make sure we are comparing exactly the same thing, we implement this 
# from scratch here ... 

function EF_err(sys::JuLIP.Atoms, calc)
   E = JuLIP.energy(calc, sys) 
   F = JuLIP.forces(calc, sys)
   E_ref = JuLIP.get_data(sys, "energy")
   F_ref = JuLIP.get_data(sys, "force")
   return abs(E - E_ref) / length(sys), norm.(F - F_ref)
end

function EF_err(sys::AtomsBase.AbstractSystem, calc)
   efv = M.energy_forces_virial(sys, calc_model2_fit)
   F_ustrip = [ ustrip.(f) for f in efv.forces ]
   E_ref = sys.data[:energy]
   F_ref = sys.data[:force]
   return abs(ustrip(efv.energy) - E_ref) / length(sys), norm.(F_ustrip - F_ref)
end
   
function rmse(test, calc) 
   E_errs = Float64[] 
   F_errs = Float64[] 
   for sys in test 
      E_err, F_err = EF_err(sys, calc)
      push!(E_errs, E_err)
      append!(F_errs, F_err) 
   end 
   return norm(E_errs) / sqrt(length(E_errs)), 
          norm(F_errs) / sqrt(length(F_errs))
end


E_rmse_1, F_rmse_1 = rmse(test, model1.potential)
E_rmse_2, F_rmse_2 = rmse(test2, calc_model2_fit)


@printf("Model  |     E    |    F  \n")
@printf(" ACE1  | %.2e  |  %.2e  \n", E_rmse_1, F_rmse_1)
@printf(" ACE2  | %.2e  |  %.2e  \n", E_rmse_2, F_rmse_2)

