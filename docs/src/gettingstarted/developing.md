
# Developing a new ACE1.jl model

Make sure to first read the installation notes. Now start by importing the required packages: 
```julia
using ACE1, JuLIP, IPFitting
using LinearAlgebra: norm 
```

### Step 1: specify the ACE basis 

The ACE basis can be set up using the function `ace_basis()`. 
```julia 
r0 = rnn(:Si)
basis = ace_basis(; 
      species = :Si,
      N = 3,                        # correlation order = body-order - 1
      maxdeg = 12,                  # polynomial degree
      D = SparsePSHDegree(; wL=1.5, csp=1.0),
      r0 = r0,                      # estimate for NN distance
      rin = 0.65*r0, rcut = 5.0,    # domain for radial basis (cf documentation)
      pin = 0)
@show length(basis)
```
where the parameters have the following meaning: 
* `species`: chemical species, for multiple species provide a list 
* `N` : correlation order 
* `maxdeg`: maximum polynomial degree 
* `D` : specifies the notion of polynomial degree for which there is no canonical definition in the multivariate setting. Here we use `SparsePSHDegree` which specifies a general class of sparse basis sets; see its documentation for more details.
* `r0` : an estimate on the nearest-neighbour distance for scaling, `JuLIP.rnn()` function returns element specific earest-neighbour distance
* `rin, rcut` : inner and outer cutoff radii 
* `pin` :  specifies the behaviour of the basis as the inner cutoff radius.

### Step 2: Generate a training set 

Normally one would generate a training set using DFT data, store it e.g. as 
an `.xzy` file, which can be loaded via IPFitting. Here, we will just general 
a random training set to show how it will be used. 
```julia
function gen_dat()
   sw = StillingerWeber() 
   n = rand(2:4)
   at = rattle!(bulk(:Si, cubic=true) * n, 0.3)
   return Dat(at, "diax$n"; E = energy(sw, at), F = forces(sw, at) )
end

train = [ gen_dat() for _=1:50 ]
```
* `gen_dat()` generates a single training configuration wrapped in an `IPFitting.Dat` structure. Each `d::Dat` contains the structure `d.at`, and energy value and a force vector to train against. These are stored in the dictionary `d.D`. Other observations can also be provided. The string `"diax$n"` is a configtype label given to each structure which is useful in seeing what the performance of the model is on different classes of structures. 
* `train` is then a list of 50 such training configurations.

### Step 3: Estimate Parameters 

First we evaluate the basis on all training configurations. We do this by assembling an `LsqDB` which contains all information about the basis, the training data and also stores the values of the basis on the training data for later reuse e.g. to experiment with different parameter estimation algorithms, or parameters. 
```julia 
dB = LsqDB("", basis, train)
```
Using the empty string `""` as the filename means that the `LsqDB` will not be automatically stored to disk.

To assemble the LSQ system we now need to specify weights. If we want to give the same energy and force weights to all configurations, we can just do the following: 
```julia 
weights = Dict("default" => Dict("E" => 15.0, "F" => 1.0 , "V" => 1.0 ))
```
But e.g. we could give different weights to `diax2, diax3, diax4` configs. 

Now we can fit the potential using 
```julia 
IP, lsqinfo = lsqfit(dB; weights = weights, error_table = true) 
```
This assembles the weighted LSQ system, and retuns the potential `IP` as well as a dictionary `lsqinfo` with some general information about the potential and fitting process.  E.g., to see the training errors we can use 
```julia
rmse_table(lsqinfo["errors"])
```
Note that `IP` is a `JuLIP.jl` calculator and can be used to evaluate e.g. `energy, forces, virial` on new configurations. 

### Step 4: Run some tests 

At a minimum we should have a test set to check generalisations, but more typically we would now run extensive robustness tests. For this mini-tutorial we will just implement a very basic energy generalisation test. 
```julia
test =  [ gen_dat() for _=1:20 ]
Etest = [ dat.D["E"][1] for dat in test ]
Emodel = [ energy(IP, dat.at) for dat in test ] 
rmse_E = norm(Etest - Emodel) / sqrt(length(test))
```
