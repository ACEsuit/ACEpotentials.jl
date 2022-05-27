
# ### Fitting a TiAl potential 
#
# Start by importing the required libraries 

using ACE1pack, ACE1, IPFitting, LazyArtifacts

# We need LazyArtifacts (or could also use Pkg.Artifacts instead) to obtain a dataset that is stored in ACEsuit/ACEData specifically for this tutorial. The following line will download the dataset, store is somewhere inside `~/.julia/...` and return a string with the absolute path to the file.

data_file = joinpath(artifact"TiAl_tutorial", "TiAl_tutorial.xyz")

# We can now use `IPFitting.Data.read_xyz` to load in the training set. This will not only load the structures, but also search for energies and force from a reference model, and all this will then be stored as a `Vector{Dat}`. We keep only every 10 training structures to keep the regression problem small.

data = IPFitting.Data.read_xyz(data_file, energy_key="energy", force_key="force", virial_key="virial")
train = data[1:5:end];

# The next step is to generate a basis set:  
# * Here we take 3-correlation, i.e. a 4-body potential, 
# * a relatively low polynomial degree `maxdeg = 6`, and 
# * a cutoff radious `rcut = 5.5`
# These three are the most important approximation parameters to explore when trying to improve the fit-accuracy. In addition there is
# * The parameter `r0` is just a scalig parameter and the fits should not be very sensitive to its choice. A rough estimate for the nearest-neighbour distance is usually ok. 
# * The inner cutoff `rin` is will ensure that the many-body potential becomes zero when atoms get too close. The reason for this is that we usually do not have data against which to fit the potential in this deformation regime and therefore cannot make reliable predictions. Instead we will add a pair potential to model this regime below.
#

r0 = 2.88 
ACE_B = ace_basis(species = [:Ti, :Al],
                  N = 3,
                  r0 = r0,
                  rin = 0.6 * r0,
                  rcut = 5.5,
                  maxdeg = 6);

# As alluded to above, we now add a pair potential to obtain qualitatively correct repulsive behaviour for colliding atoms. The many-body basis `ACE_B` and the pair potential `Bpair` are then combined into a single basis set `B`. 

Bpair = pair_basis(species = [:Ti, :Al],
                   r0 = r0,
                   maxdeg = 6,
                   rcut = 7.0,
                   pcut = 1,
                   pin = 0)  
B = JuLIP.MLIPs.IPSuperBasis([Bpair, ACE_B]);

# The next step is to evaluate the basis on the training set. Precomputing the basis once (and possibly save it to disk) makes experimenting with different regression parameters much more efficient. This is demonstrated below by showing various different solver options. Similarly once could also explore different data weights (see `weights` below). 

dB = LsqDB("", B, train);

# `Vref` specifies a reference potential, which is subtracted from the training data and the ACE parameters are then estimates from the difference. I.e. the this reference potential will in the end be added to the ACE model. Here we just use a one-body potential i.e. a reference atom energy for each individual species. 

Vref = OneBody(:Ti => -1586.0195, :Al => -105.5954)

# The next line specifies the regression weights: in the least squares loss different observations are given different weights,
# ```math 
#   \sum_{R} \Big( w_^E_R | E(R) - y_R^E |^2
#            + w_F^R | {\rm forces}(R) - y_R^F |^2 
#            + w_V^R | {\rm virial}(R) - y_R^V |^2,
# ```
# and this is specificed via the following dictionary. The keys correspond to the `configtype` field in the `Dat` structure. 

weights = Dict(
        "FLD_TiAl" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
        "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ))

# We are finally coming to the parameter estimation. In this tutorial we provide four different algorithms to solve the LLSQ problem: a Krylov method LSQR, rank-revealing QR, `scikit-learn` BRR solver as well as `the scikit-learn` ARD solver. 
# TODO: discuss regularisation

solver_type = :lsqr
laplace_precon = true 

if solver_type == :lsqr
	solver = Dict(
        	"solver" => :lsqr,
        	"lsqr_damp" => 1e-2,
        	"lsqr_atol" => 1e-6)
elseif solver_type == :rrqr
	solver = Dict(
        	"solver" => :rrqr,
        	"rrqr_tol" => 1e-5)
elseif solver_type == :brr
	solver = Dict(
        	"solver" => :brr,
		"brr_tol" => 1e-3)
elseif solver_type == :ard 
	solver= Dict(
         	"solver" => :ard,
         	"ard_tol" => 1e-3,
         	"ard_threshold_lambda" => 10000)
end

# TODO: discuss role of regularisation 

if laplace_precon
	using LinearAlgebra
	rlap_scal = 3.0
	P = Diagonal(vcat(ACE1.scaling.(dB.basis.BB, rlap_scal)...))
	solver["P"] = P
end

# Once all the solver parameters have been determined, we use `IPFitting.lsqfit` to estimate the parameters. This routine will return the fitted interatomic potential `IP` as well as the a dictionary `lsqfit` with some information about the fitting process. 

IP, lsqinfo = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table = true);

# For example,`lsqinfo` will contain information about the fit accuracy which we can display as follows:

@info("Training Error Table")
rmse_table(lsqinfo["errors"])

# We should of course also look at test errors, which can be done as follows. Depending on the choice of solver, and solver parameters, the test errors might be very poor. Exploring different parameters in different applications can lead to significantly improved predictions. 

@info("Test Error Table")
test = data[2:10:end]
IPFitting.add_fits!(IP, test)
rmse_table(test)



# If we want to save the fitted potentials to disk to later use we can use one of the following command: the first saves the potential as an `ACE1.jl` compatible potential, while the second line exports it to a format that can be ready by the `pacemaker` code to be used within LAMMPS.
# ```julia 
#    save_dict("./TiAl_tutorial_pot.json", Dict("IP" => write_dict(IP), "info" => lsqinfo))
#    ACE1.ExportMulti.export_ACE("./TiAl_tutorial_pot.yace", IP)
# ```
