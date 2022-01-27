
# ------------------------------------------
#    Data to fit

export data_params

"""TODO: document this"""
function data_params(;
    xyz_filename = nothing,
    energy_key = "dft_energy",
    force_key = "dft_force",
    virial_key = "dft_virial"
)
    # ENH: add an alternative to specify energy/force/virial prefix only
    # TODO: replace asserts with something helpful
    @assert !isnothing(xyz_filename)

    return Dict(
        "xyz_filename" => xyz_filename,
        "energy_key" => energy_key,
        "force_key" => force_key,
        "virial_key" => virial_key
    )

end

"""TODO: document this"""
function read_data(params::Dict)
    return IPFitting.Data.read_xyz(;
        params["xyz_filename"],
        energy_key = params["energy_key"],
        force_key = params["force_key"],
        virial_key = params["virial_key"])
end

