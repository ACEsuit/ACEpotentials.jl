# This script is to roughly document how to use the new model implementations 
# I'll try to explain what can be done and what is missing along the way. 
# I am 

using ACEpotentials, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Random 

bulk = AtomsBuilder.bulk 
rattle! = AtomsBuilder.rattle!      

# because the new implementation is experimental, it is not exported, 
# so I create a little shortcut to have easy access. 

M = ACEpotentials.Models      

# The new implementation tries to follow Lux rules, which likes to be 
# disciplined and explicit about random numbers 

rng = Random.MersenneTwister(1234)


# I'll create a new model for a simple alloy and then generate a model. 
# this generates a trace-like ACE model with a random radial basis. 

elements = (:Al, :Ti)

model = M.ace_model(; elements = elements, 
                      order = 3,          # correlation order 
                      Ytype = :solid,     # solid vs spherical harmonics
                      level = M.TotalDegree(),   # how to calculate the weights to give to a basis function
                      max_level = 15,     # maximum level of the basis functions
                      pair_maxn = 15,     # maximum number of basis functions for the pair potential 
                      init_WB = :glorot_normal,     # how to initialize the ACE basis parmeters
                      init_Wpair = :glorot_normal   # how to initialize the pair potential parameters
                      )

# the radial basis specification can be looked at explicitly via 

display(model.rbasis.spec)

# we can see that it is defined as (n, l) pairs. Each `n` specifies an invariant 
# channel coupled to an `l` channel. Each `Rnl` radial basis function is defined 
# by `Rnl(r, Zi, Zj) = ∑_q W_nlq(Zi, Zj) * P_q(r)`. 

# some things that are missing: 
# - reweighting the basis via a smoothness prior. 
# - allow initialization of pair potential basis with one-hot embedding params 
#   right now the pair potential basis uses trace-like radials 
# - convenient ways to inspect the many-body basis specification. 

# Lux wants us to call a setup function to generate the parameters and state
# for the model.

ps, st = Lux.setup(rng, model)

# From the model we generate a calculator. This step should probably be integrated. 
# into `ace_model`, we can discuss it. 

calc = M.ACEPotential(model, ps, st)

# We can now treat `calc` as a nonlinear parameterized site potential model. 
# - generate a random Al, Ti structure  
# - calculate the energy, forces, and virial
# An important point to note is that AtomsBase enforces the use of units. 

function rand_AlTi(nrep, rattle)
   # Al : 13; Ti : 22
   at = rattle!(bulk(:Al, cubic=true) * 2, 0.1)
   Z = AtomsBuilder._get_atomic_numbers(at)
   Z[rand(1:length(at), length(at) ÷ 2)] .= 22
   return AtomsBuilder._set_atomic_numbers(at, Z)      
end


at = rand_AlTi(2, 0.1)

efv = M.energy_forces_virial(at, calc, ps, st)

@info("Energy")
display(efv.energy)

@info("Virial")
display(efv.virial)

@info("Forces (on atoms 1..5)")
display(efv.forces[1:5])



