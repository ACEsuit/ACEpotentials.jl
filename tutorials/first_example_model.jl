# # First example (Julia)

# ## Developing a new ACE1.jl model

# This very simple tutorial constructs an ACE1 model for Si by fitting to an empirical potential.

# Make sure to first read the installation notes. Now start by importing the required packages: 
using ACE1pack 
import Random
using LinearAlgebra: norm, Diagonal

# ### Step 1: specify the ACE Model
#
# The parameters have the following meaning: 
# * `species`: chemical species, for multiple species provide a list 
# * `N` : correlation order
# * `maxdeg`: maximum total polynomial degree 
# * `rcut` : cutoff radius

model = acemodel(species = :Si,
                 N = 3,   
                 maxdeg = 12,           
                 rcut = 5.0)
@show length(model.basis);

# ### Step 2: Generate a training set 
#
# Next we need to generate some training data to estimate the model parameters. Normally one would generate a training set using DFT data, stored as an `.xyz` file. Here, we create a random training set for simplicity. Please note that this is generally *not* a good strategy to generate data!
# * `gen_dat()` generates a single training configuration wrapped in an `ACE1pack.AtomsData` structure. Each `d::AtomsData` contains the structure `d.atoms`, and energy value and a force vector to train against. 
# * `train` is then a collection of such random training configurations.

data_keys = (energy_key = "energy", force_key = "forces")

function gen_dat()
   sw = StillingerWeber() 
   at = rattle!(bulk(:Si, cubic=true) * rand(2:3), 0.3)
   set_data!(at, data_keys.energy_key, energy(sw,at))
   set_data!(at, data_keys.force_key, forces(sw,at))
   return at
end

Random.seed!(0)
train = [gen_dat() for _=1:20];

# ### Step 3: Estimate Parameters 
#
# We specify a solver and then let `ACEfit.jl` to do all the work for us. More fine-grained control is possible; see the `ACEfit.jl` documentation.
# For sake of illustration we use a Bayesian Ridge Regression solver. This will automatically determine the regularisation for us. 

solver = ACEfit.RRQR(rtol = 1e-4)   
acefit!(model, train, solver; data_keys...)

# To see the training errors we can use 

@info("Training Errors")
ACE1pack.linear_errors(train, model; data_keys...)

# ### Step 4: Run some tests 
#
# At a minimum one should have a test set, check the errors on that test set, and confirm that they are similar as the training errors. 

@info("Test Errors")
test =  [gen_dat() for _=1:20]
ACE1pack.linear_errors(test, model; data_keys...)

# If we wanted to perform such a test ``manually'' it might look like this: 

@info("Manual RMSE Test")
potential = model.potential
test_energies = [ JuLIP.get_data(at, "energy") / length(at) for at in test]
model_energies = [energy(potential, at) / length(at) for at in test] 
rmse_energy = norm(test_energies - model_energies) / sqrt(length(test))
@show rmse_energy;

# But in practice, one should run more extensive test simulations to check how robust the fitted potential is. This is beyond the scope of this tutorial.

