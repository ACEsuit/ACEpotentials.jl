# ACE1 and ACEfit

`ACE1.jl` defines functions from the space of local atomic environments `` \mathcal{X}`` to the space of real numbers ``\mathbb{R}``. These functions respect physical symmetries such as invariance under rotation of the environment and permutation of equivalent atoms. This set of functions ``\{B_\nu \}_\nu`` may be treated as a basis of a space of such symmetric functions, allowing us to express a property of an atomic environment ``R \in \mathcal{X}`` as follows:
```math
y(R) = \sum_\nu c_\nu B_\nu(R)
```
Having explicitly constructed such a basis set, the coefficients ``c_\nu`` can be found by fitting the model to data ``\{ (R_i, y_i) \}_i`` and solving by, for instance, least squares:
```math
\mathbf{c} = \text{arg} \min_\mathbf{c} \sum_i \left( y_i - \sum_\nu c_\nu B_\nu(R_i) \right)^2 + \text{REG}
```
`ACE1.jl` describes the symmetric basis set; `ACE1pack.jl` and `ACEfit.jl` handle the assembly and solution of the resulting least squares system, and provides a variety of methods for doing so including different regularization methods.
`ACEfit.jl` also defines a generic `Data` type and `ACE1pack.jl` implements a version of this type which represents a labelled atomic configuration.

# The Least Squares Database

The minimisation problem above can be written:
```math
\mathbf{c} = \text{arg} \min_\mathbf{c} \| \mathbf{y} - \Psi \mathbf{c} \|^2
```
where ``y_i`` are the observations of the true function, and ``\Psi_{i \nu} = B_\nu(R_i)`` is the design matrix. `ACE1pack` and `ACEfit` construct the design matrix and the observation vector (from the basis and training configurations) and stores them in a database: [[source]](https://github.com/ACEsuit/IPFitting.jl/blob/main/src/lsq_db.jl)
```julia
dB = LsqDB(save_name, basis, train)
```
If `save_name` is the empty string, the least squares system, which can be very large, is not saved to disk. Otherwise, `save_name` should be a string not including any file extension, which is added by... `basis` is the ACE1 basis. `train` is a `Vector` of ACE1pack `AtomsData` objects representing the training set. TODO also provides tools for reading and saving the atomic structures see [File io (TO-ADD, currently points to ACE1pack data handling)](../ACE1pack/data.md). 

### Structure of the Linear System.

Observations of the energy, forces and virial stresses of an atomic configuration can be used to train a model. Each scalar observable contributes one row to the linear system: An energy observation therefore contributes a single row, and the forces on all the of the N atoms in a configuration contribute 3N rows.
Training configurations can also be distinguised from one another by setting the `configtype` field in the ACE1pack `AtomsData` object. The least squares database recognises the config type of a configuration, which can be used to apply different settings for different config types when fitting.

# Solving the Linear System

Fitting is performed by calling `lsqfit`.
```julia
IP, lsqinfo = lsqfit(dB, solver=solver, weights=weights, Vref=Vref, error_table=true)
```
arguments (see below for details):
* dB : `IPFitting.lsqDB`. The least sqaures system to be solved.
* solver : `Dict()`. Specifies the solution method.
* weights : `Dict()`. The weights of the different observations (rows) of the least squares system.
* Vref : `Dict()`. A reference potential.
* error_table : `bool` If `true` a table containing fitting errors is printed and stored in `lsqinfo`. 
returns:
* IP : The interatomic potential that can be evaluated on a new configuration.
* lsqinfo : A dictionary of information about the least sqaures system and the solution process.

### Solvers

Once the linear system has been formed, several methods exist for solving it. Some involve modifying the above minimisation statement but still require the design matrix and observation vector. Currently there are 4 solvers implemented in IPFitting which are discussed in [solvers](Solvers.md).

### Weights

The weights dictionary can be used to rescale rows of the linear equation to emphasise some observations more than others. For instance, it may be useful to weight the rows of the linear system corresponding to the energy larger than those corresponding to forces, if there are many more force observations than energy observations.
Different weight can also be set for different config types. An example for a database containing training data with config types `MD` and `Phonon`, might be:
```julia
weights = Dict(
        "MD" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ),
        "Phonon" => Dict("E" => 10.0, "F" => 10.0 , "V" => 10.0 ))
```

### Reference potential

It is also possible to suply a reference potential ``V``, which acts as a baseline for the prediction. If a reference potential is supplied, the prediction is modelled as
```math
y(R) = V(R) + \sum_\nu c_\nu B_\nu(R)
```
To implement this, the least squares database subtracts the reference from the observations before forming the linear system. The energy of the uninteracting isolated atoms (a `OneBody` potential) is good reference potential:
```julia
Vref = OneBody(:Ti => -1586.0195, :Al => -105.5954)
```

