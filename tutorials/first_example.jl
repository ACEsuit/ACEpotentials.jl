# # First example (Julia)

# ## Developing a new ACE1.jl model

# This is a very tutorial to demonstrate how to use ACE1 and ACEfit to construct an ACE1 model for Si by fitting to an empirical potential.

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
      maxdeg = 12,                  # polynomial degree
      D = SparsePSHDegree(; wL=1.5, csp=1.0),
      r0 = r0,                      # estimate for NN distance
      rin = 0.65*r0, rcut = 5.0,    # domain for radial basis (cf documentation)
      pin = 2)
@show length(basis)


# ### Step 2: Generate a training set 
#
# Normally one would generate a training set using DFT data, store it e.g. as an `.xyz` file, which can be loaded via ACEfit. Here, we will just general a random training set to show how it will be used. 
# * `gen_dat()` generates a single training configuration wrapped in an `ACE1pack.AtomsData` structure. Each `d::Dat` contains the structure `d.at`, and energy value and a force vector to train against. These are stored in the dictionary `d.D`. Other observations can also be provided. The string `"diax$n"` is a configtype label given to each structure which is useful in seeing what the performance of the model is on different classes of structures. 
# * `train` is then a list of 50 such training configurations.

function gen_dat()
   sw = StillingerWeber() 
   n = rand(2:4)
   at = rattle!(bulk(:Si, cubic=true) * n, 0.3)
   set_data!(at, "energy", energy(sw,at))
   set_data!(at, "forces", forces(sw,at))
   # TODO: Revise so these hacks aren't required. Also update description of weights.
   # To assemble the LSQ system we now need to specify weights. If we want to give the same energy and force weights to all configurations, we can just do the following. But e.g. we could give different weights to `diax2, diax3, diax4` configs (more on this in other tutorials).
   weights = Dict("default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ))
   v_ref = OneBody("Si" => 0.0)
   # END TODO
   return ACE1pack.AtomsData(at, "energy", "forces", "no_virial_key", weights, v_ref)
end

Random.seed!(0)
train = [ gen_dat() for _=1:50 ];


# ### Step 3: Estimate Parameters 
#
# First we assemble the design matrix (A), target vector (Y), and weight vector (W).

A, Y, W = ACEfit.linear_assemble(train, basis)

# Now we can fit the potential using 

solver = ACEfit.create_solver(Dict("type" => "qr"))
solution = ACEfit.linear_solve(solver, A, Y)
IP = JuLIP.MLIPs.combine(basis, solution["C"])

# To see the training errors we can use 

@show ACE1pack.linear_errors(train, IP)

# Note that `IP` is a `JuLIP.jl` calculator and can be used to evaluate e.g. `energy, forces, virial` on new configurations. 

# ### Step 4: Run some tests 
#
# At a minimum we should have a test set to check generalisations, but more typically we would now run extensive robustness tests. For this mini-tutorial we will just implement a very basic energy generalisation test. 

test =  [ gen_dat() for _=1:20 ]
Etest = [ d.atoms.data["energy"].data/length(d.atoms) for d in test ]
Emodel = [ energy(IP, d.atoms)/length(d.atoms) for d in test ] 
rmse_E = norm(Etest - Emodel) / sqrt(length(test))
@show rmse_E;    # rmse_E = 5.842000599246454e-5
