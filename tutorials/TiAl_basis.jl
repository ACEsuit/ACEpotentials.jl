# # The `ace_basis` interface from Julia (TiAl)
#
# Start by importing the required libraries 

using ACEpotentials

# We need a dataset `TiAl_tutorial.xyz` for this tutorial. Normally we would get the path to a datset and then use `read_extxyz` to load in the training set. 
# ```julia
# data_file = "path/to/TiAl_tutorial.xyz"
# data = read_extxyz(data_file)
# ```
# For convenience we provide this dataset as a [Julia artifact](https://docs.julialang.org/en/v1/stdlib/Artifacts/) and make it conveniently accessible via `ACEpotentials.example_dataset`. We keep only a small subset of the training structures to keep the regression problem small.

data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
train_data = data[1:5:end];

# The next step is to generate a basis set:  
# * `order = 3` : We take 3-correlation, i.e. a 4-body potential, 
# * `totaldegree = 6` : a very low polynomial degree just for testing 
# * `rcut = 5.5` : this is a typical cutoff radius, there is also a good default which is a bit higher
# These three are the most important approximation parameters to explore when trying to improve the fit-accuracy. In addition there is
# * The parameter `r0` is just a scaling parameter and the fits should not be very sensitive to its choice. A rough estimate for the nearest-neighbour distance is usually ok. (NB: if you change to a non-trivial distance transform, then the parameter `r0` may become important.)

r0 = 2.88 
basis = ACE1x.ace_basis(elements = [:Ti, :Al],
                  order = 3,
                  totaldegree = 6, 
                  rcut = 5.5, 
                  r0 = r0,)
@show length(basis);

# `Vref` specifies a reference potential, which is subtracted from the training data and the ACE parameters are then estimated from the difference. This reference potential will in the end be added to the ACE model. Here we use a one-body potential i.e. a reference atom energy for each individual species. Usage of a one-body reference potential generally results in very slightly reduced fit accuracy but significantly improved 2-body potentials with a realistic dimer shape. 

Vref = OneBody(:Ti => -1586.0195, :Al => -105.5954);

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

# The next step is to evaluate the basis on the training set. Precomputing the basis once (and possibly save it to disk) makes experimenting with different regression parameters much more efficient. This is demonstrated below by showing various different solver options. Similarly once could also explore different data weights (see `weights` below). 

datakeys = (energy_key = "energy", force_key = "force", virial_key = "virial")
train = [ACEpotentials.AtomsData(t; weights=weights, v_ref=Vref, datakeys...) for t in train_data] 
A, Y, W = ACEfit.assemble(train, basis);

# ACE1.jl has a heuristic smoothness prior built in which assigns to each basis function `Bi` a scaling parameter `si` that estimates how "rough" that basis function is. The following line generates a regularizer (prior) with `si^q` on the diagonal, thus penalizing rougher basis functions and enforcing a smoother fitted potential. 

P = smoothness_prior(basis; p = 4)

# Once all the solver parameters have been determined, we use `ACEfit` to estimate the parameters. This routine will return the fitted interatomic potential `IP` as well as the a dictionary `lsqfit` with some information about the fitting process. 

solver = ACEfit.LSQR(damp = 1e-2, atol = 1e-6, P = P)
results = ACEfit.solve(solver, W .* A, W .* Y)
pot_1 = JuLIP.MLIPs.SumIP(Vref, JuLIP.MLIPs.combine(basis, results["C"]))

# The advantage of working with the ACE basis rather than the ACE model interface is that we can now make some changes to the fitting parameters and refit. For example, we might want different weights, change the smoothness prior, and switch to a RRQR solver. 

weights["FLD_TiAl"]["E"] = 20.0
W = ACEfit.assemble_weights(train)
solver = ACEfit.RRQR(; rtol = 1e-8, P = smoothness_prior(basis; p = 2))
results = ACEfit.solve(solver, W .* A, W .* Y)
pot_2 = JuLIP.MLIPs.SumIP(Vref, JuLIP.MLIPs.combine(basis, results["C"]))


# We can now compare the errors in a nice table. Depending on the choice of solver, and solver parameters, the test errors might be very poor. Exploring different parameters in different applications can lead to significantly improved predictions. 

test = [ACEpotentials.AtomsData(t; weights=weights, v_ref=Vref, datakeys...) for t in data[2:10:end]] 

@info("Test Error Tables")
@info("First Potential: ")
ACEpotentials.linear_errors(test, pot_1);

@info("Second Potential: ")
ACEpotentials.linear_errors(test, pot_2);

# If we want to save the fitted potentials to disk to later use we can use one of the following commands: the first saves the potential as an `ACE1.jl` compatible potential, while the second line exports it to a format that can be ready by the `pacemaker` code to be used within LAMMPS.

save_dict("./TiAl_tutorial_pot.json", Dict("IP" => write_dict(pot_1)))

# The fitted potential can also be exported to a format compatible with LAMMPS (ML-PACE)

export2lammps("./TiAl_tutorial_pot.yace", pot_1)

