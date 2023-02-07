
# ------------------------------------------
#    Data to fit

export data_params

"""
data_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to read data from an .xyz file. j 
All parameters are passed as keyword arguments.

### Parameters
* `fname` : a "*.xyz" file with atomistic data (mandatory).
* `energy_key = "energy` : ASE's `Atoms.info` key to read energy 
for fitting.
* `force_key = "force` : ASE's `Atoms.arrays` key to read forces 
for fitting.
* `virial_key = "virial` : ASE's `Atoms.info` key to read virial 
for fitting.
* `weight_key = "config_type"`: ASE's `Atoms.info` key used for picking the correct weight out of the weights dictionary. 
"""
function data_params(;
    fname = nothing,
    energy_key = "energy",
    force_key = "force",
    virial_key = "virial",
    weight_key = "config_type"
)
    @assert !isnothing(fname) "`fname` must be given. "

    return Dict(
        "fname" => fname,
        "energy_key" => energy_key,
        "force_key" => force_key,
        "virial_key" => virial_key,
        "weight_key" => weight_key
    )

end
