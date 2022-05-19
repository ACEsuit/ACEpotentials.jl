
# ------------------------------------------
#   Read ACE fit parameters from file 

using ACE1pack

export  fill_defaults!

""" Recursively updates nested dictionaries with default parameters"""
function fill_defaults!(params::Dict; param_key = "fit_params")
    # Go through the nested dictionaries filling in the default values
    params = _fill_default(params, param_key)
    for (key, val) in params
        if key == "basis"
            for (basis_name, basis_params) in params[key]
                params[key][basis_name] = fill_defaults!(basis_params; param_key=basis_name)
            end
        elseif key == "transforms"
            params[key] = _fill_transforms(val)
        elseif val isa Dict && haskey(_dict_constructors, key)
            params[key] = fill_defaults!(val; param_key = key)
        end
    end
    return params
end

_fill_transforms(params) = Dict([key => fill_defaults!(val, param_key="transform") for (key, val) in params])
_fill_default(d::Dict, key) = _dict_constructors[key](;_makesymbol(d)...)
_makesymbol(p::Pair) = (Symbol(p.first) => (p.second))
_makesymbol(D::Dict) = Dict(_makesymbol.([D...])...)   

_dict_constructors = Dict(
    "fit_params" => ACE1pack.fit_params,
    "data" => ACE1pack.data_params,
    "solver" => ACE1pack.solver_params,
    "rad_basis" => ACE1pack.rad_basis_params,
    "rpi_basis" => ACE1pack.rpi_basis_params,
    "pair_basis" => ACE1pack.pair_basis_params,
    "transform" => ACE1pack.transform_params, 
    "degree" => ACE1pack.degree_params,
    "P" => ACE1pack.precon_params)

