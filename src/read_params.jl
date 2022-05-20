
# ------------------------------------------
#   Read ACE fit parameters from file 

using ACE1pack

export  fill_defaults!, parse_basis_keys

""" Recursively updates nested dictionaries with default parameters"""
function fill_defaults!(params::Dict; param_key = "fit_params")
    # Go through the nested dictionaries filling in the default values
    params = _fill_default(params, param_key)
    for (key, val) in params
        if key == "basis"
            for (basis_name, basis_params) in params[key]
                if basis_name == "rpi"
                    basis_params = parse_basis_keys(basis_param)
                end
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


"""
`("C", "C")`-type tuples are saved to and read back in as `"(\"C\", \"C\")"` to .json. 
It's slightly easier to save these to .json or .yaml as `"(C, C)"`. 
This function converts `"(C, C)"`-type strings back to parameter-friendly `("C", "C")`. 

A bit ugly, but to change the dictionary keys (and so dictionary type) one needs to construct 
it from scratch, at least as far as I can tell. 
"""
function parse_basis_keys(rpi_basis::Dict)
    basis_out = Dict()
    for (key, val) in rpi_basis
        if val isa Dict && key in ["transform", "degree"] 
            sub_dict = Dict()
            for (sub_key, sub_val) in val 
                if sub_key in ["transforms", "Dn", "Dl", "Dd"]
                    sub_dict[sub_key] = _parse_keys(sub_val)
                elseif sub_key == "cutoffs"
                    sub_dict[sub_key] = _parse_keys_and_vals(sub_val)
                else
                    sub_dict[sub_key] = sub_val
                end
            end
            basis_out[key] = sub_dict
        else
            basis_out[key] = val
        end
    end
    return basis_out
end


_parse_keys_and_vals(dict::Dict) = _parse_vals(_parse_keys(dict))

_parse_vals(dict::Dict) = 
    Dict(_is_string_of_tuple(val) ? key => _parse_string_into_tuple(val) : key => val for (key, val) in dict)

_parse_keys(dict::Dict) = 
    Dict(_is_string_of_tuple(key) ? _parse_string_into_tuple(key) => val : key => val for (key, val) in dict)

function _is_string_of_tuple(string)
    if !(typeof(string) <: AbstractString)
        return false
    end
    if string[1] == '(' && string[end] == ')'
        return true
    end
    return false
end

function _parse_entry(input::AbstractString)
    for T in [Int, Float32]
        try
            entry = parse(T, input)
            return entry
        catch
        end
    end
    return input 
end

function _parse_string_into_tuple(input::AbstractString)
    input = strip(input, ['(', ')'])
    entries = [strip(s) for s in split(input, ",")]
    return Tuple(_parse_entry(String(entry)) for entry in entries)
end

