# # Fitting a TiAl potential (Julia) - `acebasis`
#
# Start by importing the required libraries 

using ACE1pack

# We need a dataset `TiAl_tutorial.xyz` for this tutorial which is provided as an artifact. Normally we would get the path to a datset via `artifact"TiAl_tutorial` but for these tutorial to run from anywhere it is easiest to let `ACE1pack` load the data for us. The following line will download the dataset, store is somewhere inside `~/.julia/...` and return a string with the absolute path to the file.

data_file = joinpath(ACE1pack.artifact("TiAl_tutorial"), "TiAl_tutorial.xyz")

# We can now use `JuLIP.read_extxyz` to load in the training set. We keep only a small subset of the training structures to keep the regression problem small.

data = JuLIP.read_extxyz(data_file)
train_data = data[1:5:end];

# The next step is to generate a basis set:  
# * `N = 3` : We take 3-correlation, i.e. a 4-body potential, 
# * `maxdeg = 6` : a very low polynomial degree just for testing 
# * `rcut = 5.5` : this is a typical cutoff radius
# These three are the most important approximation parameters to explore when trying to improve the fit-accuracy. In addition there is
# * The parameter `r0` is just a scaling parameter and the fits should not be very sensitive to its choice. A rough estimate for the nearest-neighbour distance is usually ok. (NB: if you change to a non-trivial distance transform, then the parameter `r0` may become important.)
# * The inner cutoff `rin` with `pin = 2` results in an envelope for the radial basis that becomes zero when atoms get too close. The reason for this is that we usually do not have data against which to fit the potential in this deformation regime and therefore cannot make reliable predictions. Instead we will add a pair potential to model this regime below.

r0 = 2.88 
ACE_B = ace_basis(species = [:Ti, :Al],
                  N = 3,
                  maxdeg = 6, 
                  rcut = 5.5, 
                  r0 = r0,
                  rin = 0.6 * r0, pin = 2);

# As alluded to above, we now add a pair potential to obtain qualitatively correct repulsive behaviour for colliding atoms. The many-body basis `ACE_B` and the pair potential `Bpair` are then combined into a single basis set `B`. 

Bpair = pair_basis(species = [:Ti, :Al],
                   r0 = r0,
                   maxdeg = 6,
                   rcut = 7.0)  

# We combine the two bases into a superbasis. 

B = JuLIP.MLIPs.IPSuperBasis([Bpair, ACE_B]);

# `Vref` specifies a reference potential, which is subtracted from the training data and the ACE parameters are then estimated from the difference. This reference potential will in the end be added to the ACE model. Here we use a one-body potential i.e. a reference atom energy for each individual species. Usage of a one-body reference potential generally results in very slightly reduced fit accuracy but significantly improved 2-body potentials with a realistic dimer shape. 

Vref = OneBody(:Ti => -1586.0195, :Al => -105.5954)

# The next line specifies the regression weights: in the least squares loss different observations are given different weights,
# ```math 
#   \sum_{R} \Big( w_^E_R | E(R) - y_R^E |^2
#            + w_F^R | {\rm forces}(R) - y_R^F |^2 
#            + w_V^R | {\rm virial}(R) - y_R^V |^2 \Big),
# ```
# and this is specificed via the following dictionary. The keys correspond to the `config_type` of the training structures. 

weights = Dict(
        "FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0 , "V" => 1.0 ),
        "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ))

# The next step is to evaluate the basis on the training set. Precomputing the basis once (and possibly save it to disk) makes experimenting with different regression parameters much more efficient. This is demonstrated below by showing various different solver options. Similarly once could also explore different data weights (see `weights` below). 

train = [ACE1pack.AtomsData(t,"energy","force","virial",weights,Vref) for t in train_data] 
A, Y, W = ACEfit.linear_assemble(train, B)

# We are finally coming to the parameter estimation. In this tutorial we provide four different algorithms to solve the LLSQ problem: a Krylov method LSQR, rank-revealing QR, `scikit-learn` BRR solver as well as `the scikit-learn` ARD solver. 

solver_type = :lsqr
smoothness_prior = true 

if solver_type == :lsqr
	solver = Dict(
        	"type" => "LSQR",
        	"damp" => 1e-2,
        	"atol" => 1e-6)
elseif solver_type == :rrqr
	solver = Dict(
        	"type" => "RRQR",
        	"tol" => 1e-5)
elseif solver_type == :brr
	solver = Dict(
        	"type" => "BRR",
		"tol" => 1e-3)
elseif solver_type == :ard 
	solver= Dict(
         	"type" => "ARD",
         	"tol" => 1e-3,
         	"threshold_lambda" => 10000)
end

# ACE1.jl has a heuristic smoothness prior built in which assigns to each basis function `Bi` a scaling parameter `si` that estimates how "rough" that basis function is. The following line generates a regularizer (prior) with `si^q` on the diagonal, thus penalizing rougher basis functions and enforcing a smoother fitted potential. 

if smoothness_prior
    using LinearAlgebra
    q = 3.0
    solver["P"] = Diagonal(vcat(ACE1.scaling.(B.BB, q)...))
end

# Once all the solver parameters have been determined, we use `ACEfit` to estimate the parameters. This routine will return the fitted interatomic potential `IP` as well as the a dictionary `lsqfit` with some information about the fitting process. 

solver = ACEfit.create_solver(solver)
results = ACEfit.linear_solve(solver, A, Y)
potential = JuLIP.MLIPs.SumIP(Vref, JuLIP.MLIPs.combine(B, results["C"]))

# We can display an error table as follows:

@info("Training Error Table")
ACE1pack.linear_errors(train, potential);

# We should of course also look at test errors, which can be done as follows. Depending on the choice of solver, and solver parameters, the test errors might be very poor. Exploring different parameters in different applications can lead to significantly improved predictions. 

@info("Test Error Table")
test = [ACE1pack.AtomsData(d,"energy","force","virial",weights,Vref) for d in data[2:10:end]] 
ACE1pack.linear_errors(test, potential);

# If we want to save the fitted potentials to disk to later use we can use one of the following commands: the first saves the potential as an `ACE1.jl` compatible potential, while the second line exports it to a format that can be ready by the `pacemaker` code to be used within LAMMPS.

save_dict("./TiAl_tutorial_pot.json", Dict("IP" => write_dict(potential)))
ACE1pack.ExportMulti.export_ACE("./TiAl_tutorial_pot.yace", potential; export_pairpot_as_table=true)
