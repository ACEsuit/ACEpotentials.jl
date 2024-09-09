using Pkg

export load_potential
export save_potential

"""
    function load_potential(fname::AbstractString; new_format=false, verbose=true)

Load ACE potential from given file `fname`.

# Kwargs
- `new_format=false` - If true returns potential as `ACEmd.ACEpotential` format, else use old JuLIP format
- `verbose=true`     - Display version info on load
"""
function load_potential(
    fname::AbstractString;
    new_format=false,
    verbose=true
)
    pot_tmp = load_dict(fname)
    if verbose && haskey(pot_tmp, "Versions")
        println("\nThis potential was saved with following versions:\n")
        for (k,v) in pot_tmp["Versions"]
            n = VersionNumber(v["major"], v["minor"], v["patch"])
            println(k," v",n)
        end
        println("\n", "If you have problems with using this potential, pin your installation to above versions.\n")
    end
    if haskey(pot_tmp, "IP")
        pot = read_dict(pot_tmp["IP"])
    elseif haskey(pot_tmp, "potential")
        pot = read_dict(pot_tmp["potential"])
    else
        error("Potential format not recognised")
    end
    if new_format
        return ACEpotential(pot.components)
    else
        return pot
    end
end


"""
    save_potential( fname, potential::ACE1x.ACE1Model; save_version_numbers=true, meta=nothing)

Save ACE potentials. Prefix is either .json, .yml or .yace, which also determines file format.

# Kwargs
- save_version_numbers=true  : If true save version information or relevant packages
- `meta=nothing`             : Seve some metadata with the potential (needs to be `Dict{String, Any}`)
"""
function save_potential( fname, potential::ACE1x.ACE1Model; save_version_numbers=true, meta=nothing)
    return save_potential(fname, potential.potential; save_version_numbers=save_version_numbers, meta=meta)
end

function save_potential( fname, potential::ACEmd.ACEpotential; save_version_numbers=true, meta=nothing)
    return save_potential(fname, potential.potentials; save_version_numbers=save_version_numbers, meta=meta)
end

function save_potential(fname, potential; save_version_numbers=true, meta=nothing)
    if save_version_numbers
        versions = Dict()
        versions["ACEpotentials"] = extract_version("ACEpotentials")
        versions["ACEfit"] = extract_version("ACEfit")

        data = Dict(
            "IP" => write_dict(potential),
            "Versions" => versions
        )
    else
        data = Dict(
            "IP" => write_dict(potential)
        )
    end
    if !isnothing(meta)
        @show meta 
        @assert isa(meta, Dict{String, <:Any}) "meta needs to be a Dict{String, Any}"
        data["meta"] = convert(Dict{String, Any}, meta)
    end
    save_dict(fname, data)
end


# used to extraction version numbers when saving
function extract_version(name::AbstractString)
    try
        vals = Pkg.dependencies()|> values |> collect
        hit = filter(x->x.name==name, vals) |> only
        return hit.version
    catch 
        try 
            if name == Pkg.project().name 
                return Pkg.project().version 
            end
        catch 
            @error("Couldn't determine version of $name")
            return v"0.0.0"
        end
    end 
end


## Deprecations

@deprecate export2json(fname, model; meta=nothing) save_potential(fname, model; meta=meta)

