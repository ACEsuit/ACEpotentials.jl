import ArgParse 

import .ACE1compat
using ACEfit
using TOML

# === nt utilities ===
function create_namedtuple(dict)
   return NamedTuple{Tuple(Symbol.(keys(dict)))}(values(dict))
end

function nested_namedtuple_to_dict(nt)
   return Dict(k => isa(v, NamedTuple) ? nested_namedtuple_to_dict(v) : v for (k, v) in pairs(nt))
end
 
function _sanitize_arg(arg)
   if isa(arg, Vector)  
      return _sanitize_arg.(tuple(arg...))
   elseif isa(arg, String)
      return Symbol(arg)
   else
      return arg
   end
end

function _sanitize_dict(dict)
   return Dict(Symbol(key) => _sanitize_arg(dict[key]) for key in keys(dict))
end

##

# === make fits === 

""" 
      make_model(model_dict::Dict) 

User-facing script to generate a model from a dictionary. See documentation 
for details.       
"""
function make_model(model_dict::Dict)
   if model_dict["model_name"] == "ACE1"
      model_nt = _sanitize_dict(model_dict)
      return ACE1compat.ace1_model(; model_nt...)
   else
      error("Unknown model: $(fitting_params["model"]["model_name"]). This function only supports 'ACE1'.")
   end
end

# chho: make this support other solvers
function make_solver(solver_dict::Dict)
   if solver_dict["name"] == "BLR"
      params_nt = _sanitize_dict(solver_dict["param"])
      return ACEfit.BLR(; params_nt...)
   elseif solver_dict["name"] == "LSQR"
      params_nt = _sanitize_dict(solver_dict["param"])
      return ACEfit.LSQR(; params_nt...)
   else
      error("Not implemented.")
   end
end

# function make_prior(model, prior_dict::Dict)
#    if prior_dict["name"] === "algebraic"
#       return ACEpotentials.Models.algebraic_smoothness_prior(model.basis; p = prior_dict["param"])
#    else
#       error("Not implemented.")
#    end
# end

"""
      copy_runfit(dest)

Copies the `runfit.jl` script and an example model parameter file to `dest`.
If called from the destination directory, use 
```julia
ACEpotentials.copy_runfit(@__DIR__())
```
This is intended to setup a local project directory with the necessary 
scripts to run a fitting job.
"""
function copy_runfit(dest)
   script_path = joinpath(@__DIR__(), "..", "scripts")
   runfit_orig = joinpath(script_path, "runfit.jl")
   exjson_orig = joinpath(script_path, "example_params.json")
   runfit_dest = joinpath(dest, "runfit.jl")
   exjson_dest = joinpath(dest, "example_params.json")
   run(`cp $runfit_orig $runfit_dest`)
   run(`cp $exjson_orig $exjson_dest`)
   return nothing
end


"""
      save_model(model, filename; kwargs...) 

save model constructor, model parameters, and other information to a JSON file. 

* `model` : the model to be saved
* `filename` : the name of the file to which the model will be saved
* `make_model_args` : the arguments used to construct the model; without this 
            the model cannot be reconstructed unless the original script is available
* `errors` : the fitting / test errors computed during the fitting 
* `verbose` : print information about the saving process     
"""


function find_manifest(start_path = pwd())
    current_path = start_path
    while !isempty(current_path)
        manifest_path = joinpath(current_path, "Manifest.toml")
        if isfile(manifest_path)
            return manifest_path
        end

        current_path = dirname(current_path)
        if current_path == "/"
            break  
        end
    end
    return nothing
end

function save_model(model, filename;
                    make_model_args = nothing,
                    errors = nothing,
                    verbose = true,
                    meta = Dict())

    D = Dict("model_parameters" => model.ps,
             "meta" => meta)

    if isnothing(make_model_args)
        if verbose
            @warn("Only model parameters are saved but no information to reconstruct the model.")
        end
    else
        D["make_model_args"] = make_model_args
    end

    if !isnothing(errors)
        D["errors"] = errors
    end

    manifest_path = find_manifest()
    if manifest_path !== nothing
        try
            manifest_content = TOML.parsefile(manifest_path)
            D["manifest"] = manifest_content
        catch e
            @warn("Failed to read Manifest.toml: $e")
        end
    else
        @warn("Manifest.toml file not found in the project directory or any parent directories.")
    end

    open(filename, "w") do io
        write(io, JSON.json(D, 3))
    end

    if verbose
        @info "Results saved to file: $filename"
    end
end
