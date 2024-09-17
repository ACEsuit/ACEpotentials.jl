# # First example

# This very simple tutorial constructs an ACE1 model for Si by fitting to an empirical potential.

# Make sure to first read the installation notes. Now start by importing the required packages: 
using ACEpotentials 
import Random
using LinearAlgebra: norm, Diagonal

# ### Step 1: specify the ACE Model
#
# The parameters have the following meaning: 
# * `elements`: list of chemical species, symbols 
# * `order` : correlation order
# * `totaldegree`: maximum total polynomial degree used for the basis 
# * `rcut` : cutoff radius (optional, defaults are provided)

model = acemodel(elements = [:Si,], 
                 order = 3,   
                 totaldegree = 10,           
                 rcut = 5.0)
@show length(model.basis);

# ### Step 2: Generate a training set 
#
# Next we need to generate some training data to estimate the model parameters. Normally one would generate a training set using DFT data, stored as an `.xyz` file. Here, we create a random training set for simplicity. Please note that this is generally *not* a good strategy to generate data!
# * `gen_dat()` generates a single training configuration wrapped in an `ACEpotentials.AtomsData` structure. Each `d::AtomsData` contains the structure `d.atoms`, and energy value and a force vector to train against. 
# * `train` is then a collection of such random training configurations.

data_keys = (energy_key = "energy", force_key = "forces")

function gen_dat()
   sw = StillingerWeber() 
   at = rattle!(bulk(:Si, cubic=true) * rand(2:3), 0.3)
   set_data!(at, data_keys.energy_key, JuLIP.energy(sw,at))
   set_data!(at, data_keys.force_key, JuLIP.forces(sw,at))
   return at
end

Random.seed!(0)
train = [gen_dat() for _=1:20];

# ### Step 3: Estimate Parameters 
#
# We specify a solver and then let `ACEfit.jl` to do all the work for us. More fine-grained control is possible; see the `ACEfit.jl` documentation.
# For sake of illustration we use a Bayesian Ridge Regression solver. This will automatically determine the regularisation for us. 

solver = ACEfit.BLR() 
acefit!(model, train; solver=solver, data_keys...);

# To see the training errors we can use 

@info("Training Errors")
ACEpotentials.compute_errors(train, model; data_keys...);

# ### Step 4: Run some tests 
#
# At a minimum one should have a test set, check the errors on that test set, and confirm that they are similar as the training errors. 

@info("Test Errors")
test =  [gen_dat() for _=1:20]
ACEpotentials.compute_errors(test, model; data_keys...);

# If we wanted to perform such a test ``manually'' it might look like this: 

@info("Manual RMSE Test")
potential = model.potential
test_energies = [ JuLIP.get_data(at, "energy") / length(at) for at in test]
model_energies = [energy(potential, at) / length(at) for at in test] 
rmse_energy = norm(test_energies - model_energies) / sqrt(length(test))
@show rmse_energy;

# But in practice, one should run more extensive test simulations to check how robust the fitted potential is.

# ### Step 5: export the model 
# 
# The fitted model can be exported to a JSON or YAML file, or to a LAMMPs compatible `yace` file. We won't go through that in this tutoral. See `export2json` and `export2lammps` for further information. 

# ### Step 6: Using the model
#  Let's do something very simple: relax a vacancy. 

# We create a small Si cell, delete an atom and rattle the rest 

at = bulk(:Si, cubic=true) * 3
deleteat!(at, 1)
rattle!(at, 0.03 * rnn(:Si))

# we can now minimize the ACE energy. 

set_calculator!(at, potential);
minimise!(at)
E_ace = energy(at)

# If we want a formation energy, we could get it like this. 

at0 = bulk(:Si)
E0_ace = energy(potential, at0);
Evac_ace = E_ace - (length(at)-1)/length(at0) * E0_ace
@show Evac_ace;

# Note that there are no vacancy structures in the training set, so this is a prediction out of sample. We have no guarantee of the accuracy of this prediction. In fact the prediction is quite far off: 

sw = StillingerWeber()
set_calculator!(at, sw)
minimise!(at; verbose=false);
E_sw = energy(at)
E0_sw = energy(sw, bulk(:Si))
Evac_sw = E_sw - (length(at)-1)/length(at0) * E0_sw
@show Evac_sw;

# To obtain accurate predictions on a vacancy structure, we must add it to the training set. This iterative model development process goes beyond the scope of this tutorial.
