
# ------------------------------------------
#   Read ACE fit parameters from file 

using JSON

export json_to_params, yaml_to_params

function yaml_to_params(filename::AbstractString)
    # return ace_params()
end

function json_to_params(filename::AbstractString)
    # TODO: replace any missing values with defaults
    return JSON.parsefile(filename)
end
