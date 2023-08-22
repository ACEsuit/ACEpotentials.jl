# # The `acemodel` interface from Julia (TiAl potential)
#
# Start by importing the required libraries 

using ACEpotentials

# We need a dataset `TiAl_tutorial.xyz` for this tutorial. Normally we would get the path to a datset and then use `read_extxyz` to load in the training set. 
# ```julia
# # (don't execute this block)
# data_file = "path/to/TiAl_tutorial.xyz"
# data = read_extxyz(data_file)
# ```
# For convenience we provide this dataset as a [Julia artifact](https://docs.julialang.org/en/v1/stdlib/Artifacts/) and make it conveniently accessible via `ACEpotentials.example_dataset`. We keep only a small subset of the training structures to keep the regression problem small.

data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
train_data = data[1:5:end];

# The next step is to generate a model
# * `order = 3` : We take 3-correlation, i.e. a 4-body potential, 
# * `totaldegree = 6` : a very low polynomial degree just for testing 
# * `rcut = 5.5` : this is a typical cutoff radius for metals
# These three are the most important approximation parameters to explore when trying to improve the fit-accuracy. There are many other parameters to explore, which are documented in `?acemodel`. Even further model refinements are possible by studying the internals of `ACE1.jl` and `ACE1x.jl`.
# We also specify a reference potential that will be added to the learned 2-body and many-body potential components. Here we use a one-body potential i.e. a reference atom energy for each individual species. Usage of a one-body reference potential generally results in very slightly reduced fit accuracy but significantly improved 2-body potentials with a realistic dimer shape and improved robustness in predictions. 

model = acemodel(elements = [:Ti, :Al],
					  order = 3,
					  totaldegree = 6, 
					  rcut = 5.5, 
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

# To estimate the parameters we still need to choose a solver for the least squares system. In this tutorial we use the LSQR algorithm, for no specific reason at all. Many other solvers are available, and can be explored by looking at the documentation of `ACEfit.jl`.

solver = ACEfit.LSQR(damp = 1e-2, atol = 1e-6);

# ACE1.jl has a heuristic smoothness prior built in which assigns to each basis function `Bi` a scaling parameter `si` that estimates how "rough" that basis function is. The following line generates a regularizer (prior) with `si^q` on the diagonal, thus penalizing rougher basis functions and enforcing a smoother fitted potential. 

P = smoothness_prior(model; p = 4)    #  (p = 4 is in fact the default)

# We are now ready to estimate the parameters. We take a subset of the training data to speed up the tutorial. The prior is passed to the `acefit!` function via the `prior` keyword argument.

data_train = data[1:5:end]
acefit!(model, data_train; solver=solver, prior = P);

# We can display an error table as follows:

@info("Training Error Table")
ACEpotentials.linear_errors(data_train, model; weights=weights);

# We should of course also look at test errors, which can be done as follows. Depending on the choice of solver, and solver parameters, the test errors might be very poor. Exploring different parameters in different applications can lead to significantly improved predictions. 

@info("Test Error Table")
test_data = data[2:10:end]
ACEpotentials.linear_errors(test_data, model; weights=weights);

# If we want to save the fitted potentials to disk to later use we can use one of the following commands: the first saves the potential as an `ACE1.jl` compatible potential, while the second line exports it to a format that can be ready by the `pacemaker` code to be used within LAMMPS.

export2json("./TiAl_tutorial_pot.json", model)

# export to lammps (ML-PACE):

export2lammps("./TiAl_tutorial_pot.yace", model)

