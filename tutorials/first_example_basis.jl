# # First example (Julia) - `acebasis`

# This very simple tutorial constructs an ACE1 model for Si by fitting to an empirical potential.

# Make sure to first read the installation notes. Now start by importing the required packages: 
using ACE1pack 
import Random
using LinearAlgebra: norm, Diagonal

# ### Step 1: specify the ACE basis 
#
# The ACE basis can be set up using the function `ace_basis`, 
# where the parameters have the following meaning: 
# * `species`: chemical species, for multiple species provide a list 
# * `N` : correlation order
# * `maxdeg`: maximum total polynomial degree 
# * `r0` : an estimate on the nearest-neighbour distance for scaling, `JuLIP.rnn()` function returns element specific earest-neighbour distance
# * `rcut` : cutoff radius

basis = ace_basis(; species = :Si,
                    N = 3,   
                    maxdeg = 12,           
                    r0 = rnn(:Si),         
                    rcut = 5.0)
@show length(basis);

# ### Step 2: Generate a training set 
#
# Next we need to generate some training data to estimate the model parameters. Normally one would generate a training set using DFT data, stored as an `.xyz` file. Here, we create a random training set for simplicity. Please note that this is generally *not* a good strategy to generate data!
# * `gen_dat()` generates a single training configuration wrapped in an `ACE1pack.AtomsData` structure. Each `d::AtomsData` contains the structure `d.atoms`, and energy value and a force vector to train against. 
# * `train` is then a collection of such random training configurations.

function gen_dat()
   sw = StillingerWeber() 
   n = rand(2:3)
   at = rattle!(bulk(:Si, cubic=true) * n, 0.3)
   set_data!(at, "energy", energy(sw,at))
   set_data!(at, "forces", forces(sw,at))
   return ACE1pack.AtomsData(at, "energy", "forces")
end

Random.seed!(0)
train = [gen_dat() for _=1:20];

# ### Step 3: Estimate Parameters 
#
# We specify a solver and then as `ACEfit.jl` to do all the work for us. More fine-grained control is possible; see the `ACEfit.jl` documentation.
# For sake of illustration we use a Bayesian Ridge Regression solver. This will automatically determine the regularisation for us. 

solver = ACEfit.RRQR(rtol = 1e-4)   
solution = ACEfit.linear_fit(train, basis, solver)

# Finally, we generate the potential from the parameters. 
# Note that `potential` is a `JuLIP.jl` calculator and can be used to evaluate e.g. `energy, forces, virial` on new configurations. 

potential = JuLIP.MLIPs.combine(basis, solution["C"])

# To see the training errors we can use 

ACE1pack.linear_errors(train, potential)

# ### Step 4: Run some tests 
#
# At a minimum one should have a test set, check the errors on that test set, and confirm that they are similar as the training errors. 

test =  [gen_dat() for _=1:20]
ACE1pack.linear_errors(test, potential)

# If we wanted to perform such a test ``manually'' it might look like this: 

test_energies = [d.atoms.data["energy"].data/length(d.atoms) for d in test]
model_energies = [energy(potential, d.atoms)/length(d.atoms) for d in test] 
rmse_energy = norm(test_energies - model_energies) / sqrt(length(test))
@show rmse_energy;

# But in practice, one should run more extensive test simulations to check how robust the fitted potential is. This is beyond the scope of this tutorial.

