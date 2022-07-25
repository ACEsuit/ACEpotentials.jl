
# ------------------------------------------
#    Data to fit

export data_params

"""
data_params(; kwargs...)` : returns a dictionary containing the 
complete set of parameters required to read data from an .xyz file. j 
All parameters are passed as keyword arguments.

### Parameters
* `fname` : a "*.xyz" file with atomistic data (mandatory).
* `energy_key = "dft_energy` : ASE's `Atoms.info` key to read energy 
for fitting.
* `force_key = "dft_force` : ASE's `Atoms.arrays` key to read forces 
for fitting.
* `virial_key = "dft_virial` : ASE's `Atoms.info` key to read virial 
for fitting.
"""
function data_params(;
    fname = nothing,
    energy_key = "dft_energy",
    force_key = "dft_force",
    virial_key = "dft_virial"
)
    @assert !isnothing(fname) "`fname` must be given. "

    return Dict(
        "fname" => fname,
        "energy_key" => energy_key,
        "force_key" => force_key,
        "virial_key" => virial_key
    )

end

function read_data(params::Dict)
    julip_dataset = JuLIP.read_extxyz(params["fname"])
    return julip_dataset
end
