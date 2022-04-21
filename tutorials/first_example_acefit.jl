
# ### Developing a new ACE1.jl model

# This is a very tutorial to demonstrate how to use ACE1 and IPFitting to construct an ACE1 model for Si by fitting to an empirical potential.

# Make sure to first read the installation notes. Now start by importing the required packages: 
import ACE1pack, ACE1, Random 
using JuLIP, ACEfit
using ACE1: rpi_basis, SparsePSHDegree
using LinearAlgebra: norm 
using ACE1pack: ObsPotentialEnergy, ObsForces, ObsVirial


# #### Step 1: specify the ACE basis 
#
# The ACE basis can be set up using the function `rpi_basis`, 
# where the parameters have the following meaning: 
# * `species`: chemical species, for multiple species provide a list 
# * `N` : correlation order 
# * `maxdeg`: maximum polynomial degree 
# * `D` : specifies the notion of polynomial degree for which there is no canonical definition in the multivariate setting. Here we use `SparsePSHDegree` which specifies a general class of sparse basis sets; see its documentation for more details.
# * `r0` : an estimate on the nearest-neighbour distance for scaling, `JuLIP.rnn()` function returns element specific earest-neighbour distance
# * `rin, rcut` : inner and outer cutoff radii 
# * `pin` :  specifies the behaviour of the basis as the inner cutoff radius.

r0 = rnn(:Si)
basis = rpi_basis(; 
      species = :Si,
      N = 3,                        # correlation order = body-order - 1
      maxdeg = 12,                  # polynomial degree
      D = SparsePSHDegree(; wL=1.5, csp=1.0),
      r0 = r0,                      # estimate for NN distance
      rin = 0.65*r0, rcut = 5.0,    # domain for radial basis (cf documentation)
      pin = 0)
@show length(basis)


# #### Step 2: Generate a training set 
#
# Normally one would generate a training set using DFT data, store it e.g. as an `.xzy` file, which can be loaded via IPFitting. Here, we will just general a random training set to show how it will be used. 
# * `gen_dat()` generates a single training configuration wrapped in an `IPFitting.Dat` structure. Each `d::Dat` contains the structure `d.at`, and energy value and a force vector to train against. These are stored in the dictionary `d.D`. Other observations can also be provided. The string `"diax$n"` is a configtype label given to each structure which is useful in seeing what the performance of the model is on different classes of structures. 
# * `train` is then a list of 50 such training configurations.

function gen_dat(; weights = Dict("E" => 1.0, "F" => 1.0))
   sw = StillingerWeber() 
   n = rand(2:4)
   at = rattle!(bulk(:Si, cubic=true) * n, 0.3)
   return Dat(at, "diax$n", 
              [ eval_obs(ObsPotentialEnergy, sw, at), 
                eval_obs(ObsForces, sw, at) ] )
end

Random.seed!(0)
train = [ gen_dat() for _=1:50 ];


# #### Step 3: Estimate Parameters 
#
# Before we assemble and solve the LSQ system we first specify weights for each observation. The defaults are 1.0, but usually we want to give different weights to energies and forces. If we want to give the same energy and force weights to all configurations, we can just do the following. But e.g. we could give different weights to `diax2, diax3, diax4` configs (more on this in other tutorials).

weights = Dict("default" => Dict(ObsPotentialEnergy => 15.0, 
                                 ObsForces => 1.0, 
                                 ObsVirial => 1.0 ))
set_weights!(train, weights)


# Now we can fit the potential using `ACEfit.llsq` and then construct a potential from the parameters and the basis using `ACE1.combine` (linear combination of basis functions...). This will assemble the weighted LSQ system, then solve it using a default solver (probably QR factorisation). 

θ = ACEfit.llsq(basis, train)
acepot = ACE1.combine(basis, θ)

# Note that `acepot` is a `JuLIP.jl` calculator and can be used to evaluate e.g. `energy, forces, virial` on new configurations. 


# Next we could look at the training errors, for which we can use 
# ```
# rmse_table(lsqinfo["errors"])
# ```
# But this still needs to be implemented in ACEfit.jl 


# #### Step 4: Run some tests 
#
# At a minimum we should have a test set to check generalisations, but more typically we would now run much more extensive robustness tests. For this mini-tutorial we will just implement a very basic energy generalisation test. This is a bit "hacky" and should possibly be improved. 

test = [ gen_dat() for _=1:20 ]
Etest = [ dat.obs[1].E for dat in test ]
Emodel = [ energy(acepot, dat.config) for dat in test ] 
rmse_E = norm(Etest - Emodel) / sqrt(length(test))
@show rmse_E;  # 0.012284355

