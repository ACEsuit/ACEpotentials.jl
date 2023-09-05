
# ------------------------------------------
#    Data to fit

#export data_params

"""
data_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to read data from an .xyz file. j 
All parameters are passed as keyword arguments.

### Parameters
* `fname` : a "*.xyz" file with atomistic data (mandatory).
* `energy_key = "energy"` : key identifying energy for fitting.
* `force_key = "forces"` : key identifying forces for fitting.
* `virial_key = "virial"` : key identifying virial tensor for fitting.
* `weight_key` = "config_type` : key identifying label for setting the correct weight from weights dictionary. 
"""
function data_params(;
    fname = nothing,
    energy_key = "energy",
    force_key = "forces",
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
