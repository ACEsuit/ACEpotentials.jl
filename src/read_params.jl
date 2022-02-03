
# ------------------------------------------
#   Read ACE fit parameters from file 

using JSON, YAML, ACE1pack 

export json_to_params, yaml_to_params, fill_default_params

function yaml_to_params(filename::AbstractString)
    raw_params = YAML.load_file(filename)
    return fill_default_params(raw_params, "fit_params")
end

function json_to_params(filename::AbstractString)
    raw_params = JSON.parsefile(filename)
    return fill_default_params(raw_params, "fit_params")
end

""" Recursively updates nested dictionaries with default parameters"""
function fill_default_params(params::Dict, param_key)
    # Go through the nested dictionaries filling in the default values
    params = _fill_default(params, param_key)
    for (key, val) in params
        if val isa Dict &&  ~(key in ["weights", "e0"])
            params[key] = fill_default_params(val, key)
        end
    end
    return params
end

"""
Converts dictionary of parameters to keyword arguments and
calls ACE1pack parameter constructor functions
"""
function _fill_default(d::Dict, key)
    dict_constructor = _dict_constructors[key]
    kwargs = _params_to_kwargs(d)
    return dict_constructor(;kwargs...)
end

_dict_constructors = Dict(
    "fit_params" => ACE1pack.ace_params,
    "data" => ACE1pack.data_params,
    "solver" => ACE1pack.solver_params,
    "rpi_basis" => ACE1pack.rpi_basis_params,
    "pair_basis" => ACE1pack.pair_basis_params,
    "radbasis" => ACE1pack.radbasis_params,
    "transform" => ACE1pack.transform_params,
    "degree" => ACE1pack.degree_params,
    "P" => ACE1pack.precon_params
)

_params_to_kwargs(params::Dict) = 
    Dict([Symbol(key) => val for (key, val) in params]...)

