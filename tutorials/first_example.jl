# # First example (Julia)

# ## Developing a new ACE1.jl model

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
# * `maxdeg`: maximum polynomial degree 
# * `D` : specifies the notion of polynomial degree for which there is no canonical definition in the multivariate setting. Here we use `SparsePSHDegree` which specifies a general class of sparse basis sets; see its documentation for more details.
# * `r0` : an estimate on the nearest-neighbour distance for scaling, `JuLIP.rnn()` function returns element specific earest-neighbour distance
# * `rin, rcut` : inner and outer cutoff radii 
# * `pin` :  specifies the behaviour of the basis as the inner cutoff radius.

r0 = rnn(:Si)
basis = ace_basis(; 
      species = :Si,
      N = 3,                        # correlation order = body-order - 1
      maxdeg = 12,                  # total polynomial degree
      r0 = r0,                      # estimate for NN distance
      rin = 0.65*r0, rcut = 5.0,    # domain for radial basis (cf documentation)
      pin = 2)
@show length(basis)

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

solver = ACEfit.BR()   
solution = ACEfit.linear_fit(train, basis, solver)

# Finally, we generate the potential from the parameters. 

potential = JuLIP.MLIPs.combine(basis, solution["C"])

# To see the training errors we can use 

ACE1pack.linear_errors(train, potential)

# Note that `potential` is a `JuLIP.jl` calculator and can be used to evaluate e.g. `energy, forces, virial` on new configurations. 

# ### Step 4: Run some tests 
#
# At a minimum we should have a test set to check generalisations, but more typically we would now run extensive robustness tests. For this mini-tutorial we will just implement a very basic energy generalisation test. 

test =  [gen_dat() for _=1:20]
test_energies = [d.atoms.data["energy"].data/length(d.atoms) for d in test]
model_energies = [energy(potential, d.atoms)/length(d.atoms) for d in test] 
rmse_energy = norm(test_energies - model_energies) / sqrt(length(test))
@show rmse_energy;
