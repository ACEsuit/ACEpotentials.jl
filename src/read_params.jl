
# ------------------------------------------
#   Read ACE fit parameters from file 

using ACE1pack

export  fill_defaults!

""" Recursively updates nested dictionaries with default parameters"""
function fill_defaults!(params::Dict; param_key = "fit_params")
    # Go through the nested dictionaries filling in the default values
    params = _fill_default(params, param_key)
    for (key, val) in params
        if val isa Dict &&  ~(key in ["weights", "e0", "basis"])
            params[key] = fill_defaults!(val; param_key = key)
        elseif key == "basis"
            for (basis_name, basis_params) in params[key]
                params[key][basis_name] = fill_defaults!(basis_params; param_key=key)
            end
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
    "fit_params" => ACE1pack.fit_params,
    "data" => ACE1pack.data_params,
    "solver" => ACE1pack.solver_params,
    "basis" => ACE1pack.basis_params,
    "rad_basis" => ACE1pack.basis_params,
    "transform" => ACE1pack.transform_params,
    "degree" => ACE1pack.degree_params,
    "P" => ACE1pack.precon_params
)

_params_to_kwargs(params::Dict) = 
    Dict([Symbol(key) => val for (key, val) in params]...)

