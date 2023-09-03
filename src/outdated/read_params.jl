
# ------------------------------------------
#   Read ACE fit parameters from file 

using ACEpotentials

export  fill_defaults, parse_ace_basis_keys

""" 

`fill_defaults(params::Dict; param_key = "fit_params") -> params`

Recursively updates any missing entries with default parameters.
Accepted `param_key` values and corresponding functions: 

```julia 
    "fit_params" => ACEpotentials.fit_params,
    "data" => ACEpotentials.data_params,
    "solver" => ACEpotentials.solver_params,
    "basis" => ACEpotentials.basis_params,
    "ace" => ACEpotentials.ace_basis_params,  
    "pair" => ACEpotentials.pair_basis_params,
    "radial" => ACEpotentials.radial_basis_params,
    "transform" => ACEpotentials.transform_params, 
    "degree" => ACEpotentials.degree_params,
    "P" => ACEpotentials.regularizer_params
```

"""
function fill_defaults(params::Dict; param_key = "fit_params")
    # Go through the nested dictionaries filling in the default values
    params = _fill_default(params, param_key)
    for (key, val) in params
        if key == "basis"
            for (basis_name, basis_params) in params[key]
                if basis_params["type"] == "ace"
                    basis_params = parse_ace_basis_keys(basis_params)
                end
                params[key][basis_name] = fill_defaults(basis_params; param_key=key)
            end
        elseif key == "transforms"
            params[key] = _fill_transforms(val)
        elseif val isa Dict && haskey(_dict_constructors, key)
            params[key] = fill_defaults(val; param_key = key)
        end
    end
    return params
end

_fill_transforms(params) = Dict([key => fill_defaults(val, param_key="transform") for (key, val) in params])
_fill_default(d::Dict, key) = _dict_constructors[key](;_makesymbol(d)...)
_makesymbol(p::Pair) = (Symbol(p.first) => (p.second))
_makesymbol(D::Dict) = Dict(_makesymbol.([D...])...)   

_dict_constructors = Dict(
    "fit_params" => ACEpotentials.fit_params,
    "data" => ACEpotentials.data_params,
    "solver" => ACEpotentials.solver_params,
    "basis" => ACEpotentials.basis_params,
    "ace" => ACEpotentials.ace_basis_params,  
    "pair" => ACEpotentials.pair_basis_params,
    "radial" => ACEpotentials.radial_basis_params,
    "transform" => ACEpotentials.transform_params, 
    "degree" => ACEpotentials.degree_params,
    "P" => ACEpotentials.regularizer_params)


"""
`parse_ace_basis_keys(ace_basis::Dict) -> ace_basis`

`("C", "C")`-type tuples are saved to and read back in from JSON as `"(\"C\", \"C\")"` .json. 
It's slightly easier to save these to JSON or YAM as `"(C, C)"`. 
This function converts `"(C, C)"`-type strings back to parameter-friendly `("C", "C")`. 
"""
function parse_ace_basis_keys(ace_basis::Dict)
    # A bit ugly, but to change the dictionary keys (and so dictionary type) one needs to construct 
    # it from scratch, at least as far as I can tell. 
    basis_out = Dict()
    for (key, val) in ace_basis
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

