# # TiAl potential (ACEpotentials-julia) 
#
# In this tutorial we repeat what was done in [Fitting a TiAL potential][TiAl.md], but only using ACEpotentials. 

# `ACEpotentials.jl` has two purposes: (1) to import and re-export `ACE1.jl`, `ACEfit.jl`, `JuLIP.jl` with guaranteed version compatibility; and (2) to have several convenience wrappers for setting up the least-squares problem (`ACE1.jl` & `JuLIP.jl`) and solving it (`ACEfit.jl`). For full documentation see [ACEpotentials overview](../ACEpotentials/acepotentials_overview.md).

# First import ACEpotentials

using ACEpotentials

# First, we need to construct various parameters' dictionaries that define various aspects of fitting an ACE potential. We use various `*params()` functions that return these dictionaries and let us only specify mandatory and non-default parameter values. 

data_param_dict = data_params(
    fname = joinpath(ACEpotentials.artifact("TiAl_tutorial"), "TiAl_tutorial.xyz"), 
    energy_key = "energy",
    force_key = "force", 
    virial_key = "virial")

# N.B. there is no way to sub-select data (should there be?) and here this tutorial diverges from [Fitting a TiAL potential][TiAl.md], but the the .xyz file is small enough. 


r0 = 2.88
species = ["Ti", "Al"]     # symbols (:Ti, :Al) also work

# `basis_params` of `type="ace"` can optionally have radial part defined. 

ace_radial_params = basis_params(
    type = "radial",
    r0 = r0, 
    rin = 0.6 * r0,
    rcut = 5.5)


# Construct ACE basis 
ACE_basis_param_dict = basis_params(
    type = "ace",
    species = species,
    N = 3, 
    maxdeg = 6,
    r0 = r0, 
    radial = ace_radial_params)

# and pair basis. 

pair_basis_param_dict = basis_params(
    type = "pair",
    species = species,
    maxdeg = 6,
    r0 = r0,
    rcut = 7.0)

# The keys in the following dictionary are for reference, the basis function kind is defined by the `type` parameter. This way, it's possible to specify multiple "ACE" and/or "pair", etc basis. 

basis_param_dicts = Dict(
    "pair" => pair_basis_param_dict,
    "ace" => ACE_basis_param_dict)


# We also need to give the "isolated atom" energies that will be subtracted from total energies before the fit. 

e0 = Dict(
    "Ti" => -1586.0195,
    "Al" =>  -105.5954)


# weights are given in a dictionary as before

weights = Dict(
        "FLD_TiAl" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
        "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 ))


# The fit will be done using LSQR solver. `lsqr_atol` has a default value of 1e-6, so we can skip it here. 

solver_param_dict = solver_params(
    type = "lsqr", 
    damp = 1e-2)

# and define parameters for smoothness prior. 

smoothness_prior_params = regularizer_params(
    type = "laplacian", 
    rlap_scal = 3.0)        # default

# Finally, let's put everything together.

ace_fit_params = fit_params(
    data = data_param_dict,
    basis = basis_param_dicts,
    solver = solver_param_dict,
    e0 = e0, 
    weights = weights, 
    P = smoothness_prior_params,
    ACE_fname = "ACE.json"  # change to `nothing` if you don't want to save the potential
)

results = ACEpotentials.fit_ace(ace_fit_params)

# The potential will also be saved to the file `ACE.json` which can be read in python or julia. 
# If you want to export the potential to LAMMPS, use

## NB : export2lammps is currently broken but will be fixed urgently
