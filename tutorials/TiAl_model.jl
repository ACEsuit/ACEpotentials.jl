# # Fitting a TiAl potential (Julia) - `acemodel`
#
# Start by importing the required libraries 

using ACE1pack

# We need a dataset `TiAl_tutorial.xyz` for this tutorial which is provided as an artifact. Normally we would get the path to a datset via `artifact"TiAl_tutorial` but for these tutorial to run from anywhere it is easiest to let `ACE1pack` load the data for us. The following line will download the dataset, store is somewhere inside `~/.julia/...` and return a string with the absolute path to the file.

data_file = joinpath(ACE1pack.artifact("TiAl_tutorial"), "TiAl_tutorial.xyz")

# We can now use `JuLIP.read_extxyz` to load in the training set. We keep only a small subset of the training structures to keep the regression problem small.

data = JuLIP.read_extxyz(data_file)
train_data = data[1:5:end];

# The next step is to generate a model
# * `N = 3` : We take 3-correlation, i.e. a 4-body potential, 
# * `maxdeg = 6` : a very low polynomial degree just for testing 
# * `rcut = 5.5` : this is a typical cutoff radius for metals
# These three are the most important approximation parameters to explore when trying to improve the fit-accuracy. In addition there is
# * The parameter `r0` is just a scaling parameter and the fits should not be very sensitive to its choice. A rough estimate for the nearest-neighbour distance is usually ok. (NB: if you change to a non-trivial distance transform, then the parameter `r0` may become important.)
# * The inner cutoff `rin` with `pin = 2` results in an envelope for the radial basis that becomes zero when atoms get too close. The reason for this is that we usually do not have data against which to fit the potential in this deformation regime and therefore cannot make reliable predictions. Instead we will add a pair potential to model this regime below.
#
# Because of the inner cutoff, the potential will have no repulsive behaviour, hence we now add a pair potential to obtain qualitatively correct repulsive behaviour for colliding atoms. This is done by specifying the parameters 
# * `rcut2` : cutoff radius for pair potential 
# * `maxdeg2` : polynomial degree for pair potential
#
# Finally, we specify a reference potential that will be added to the learned 2-body and many-body potential components. Here we use a one-body potential i.e. a reference atom energy for each individual species. Usage of a one-body reference potential generally results in very slightly reduced fit accuracy but significantly improved 2-body potentials with a realistic dimer shape and improved robustness in predictions. 

r0 = 2.88 
model = acemodel(species = [:Ti, :Al],
					  ## many-body potential parameters
					  N = 3,
					  maxdeg = 6, 
					  rcut = 5.5, 
					  r0 = r0,
					  rin = 0.6 * r0, pin = 2,
					  ## pair potential parameters 
					  rcut2 = 7.0, 
					  maxdeg2 = 6,
					  ## One-body reference energies 
					  Eref = [:Ti => -1586.0195, :Al => -105.5954])
@show length(model.basis);					  

# The next line specifies the regression weights: in the least squares loss different observations are given different weights,
# ```math 
#   \sum_{R} \Big( w^E_R | E(R) - y_R^E |^2
#            + w^F_R | {\rm forces}(R) - y_R^F |^2 
#            + w^V_R | {\rm virial}(R) - y_R^V |^2 \Big),
# ```
# and this is specificed via the following dictionary. The keys correspond to the `config_type` of the training structures. 

weights = Dict(
        "FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0 , "V" => 1.0 ),
        "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ));

# To estimate the parameters we still need to choose a solver for the least squares system. In this tutorial we provide four different algorithms to solve the LLSQ problem: a Krylov method LSQR, rank-revealing QR, `scikit-learn` BRR solver as well as `the scikit-learn` ARD solver. 

solver = ACEfit.LSQR(damp = 1e-2, atol = 1e-6);
## solver = ACEfit.RRQR(rtol = 1e-5);

# ACE1.jl has a heuristic smoothness prior built in which assigns to each basis function `Bi` a scaling parameter `si` that estimates how "rough" that basis function is. The following line generates a regularizer (prior) with `si^q` on the diagonal, thus penalizing rougher basis functions and enforcing a smoother fitted potential. To use this priot, we need to re-initialize the solver with the prior as an additional argument.

P = ACE1pack.smoothness_prior(model; p = 3)
solver = ACEfit.LSQR(damp = 1e-2, atol = 1e-6, P = P);

# We are now ready to estimate the parameters. 

data_train = data[1:5:end]
acefit!(model, data_train, solver);

# We can display an error table as follows:

@info("Training Error Table")
ACE1pack.linear_errors(data_train, model; weights=weights);

# We should of course also look at test errors, which can be done as follows. Depending on the choice of solver, and solver parameters, the test errors might be very poor. Exploring different parameters in different applications can lead to significantly improved predictions. 

@info("Test Error Table")
test_data = data[2:10:end]
ACE1pack.linear_errors(test_data, model; weights=weights);

# If we want to save the fitted potentials to disk to later use we can use one of the following commands: the first saves the potential as an `ACE1.jl` compatible potential, while the second line exports it to a format that can be ready by the `pacemaker` code to be used within LAMMPS.

potential = model.potential 
save_dict("./TiAl_tutorial_pot.json", Dict("IP" => write_dict(potential)))
ACE1pack.ExportMulti.export_ACE("./TiAl_tutorial_pot.yace", potential; export_pairpot_as_table=true);
