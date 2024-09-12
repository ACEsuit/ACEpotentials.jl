import ArgParse 
using NamedTupleTools
import .ACE1compat
using ACEfit
using Optimisers: destructure

# === nt utilities ===

recursive_dict2nt(x) = x

recursive_dict2nt(D::Dict) = (;
      [ Symbol(key) => recursive_dict2nt(D[key]) for key in keys(D)]... )
 
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
function make_solver(model, solver_dict::Dict, prior_dict::Dict)
   
   # if no prior is specified, then use I as default, which is dumb
   if isempty(prior_dict)
      P = I 
   else 
      P = make_prior(model, prior_dict)
   end 

   # if no solver is specified, then use BLR as default 
   if isempty(solver_dict) 
      return BLR(; P = P) 
   end 

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

# calles into functions defined in ACEpotentials.Models
function make_prior(model, prior_dict::Dict)
   return ACEpotentials.Models.make_prior(model, namedtuple(prior_dict))
end


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
* `model_spec` : the arguments used to construct the model; without this 
            the model cannot be reconstructed unless the original script is available
* `errors` : the fitting / test errors computed during the fitting 
* `verbose` : print information about the saving process     
"""
function save_model(model, filename; 
                    model_spec = nothing, 
                    errors = nothing, 
                    verbose = true, 
                    meta = Dict(), )

   D = Dict("model_parameters" => destructure(model.ps)[1], 
            "meta" => meta)

   if isnothing(model_spec) 
      if verbose
         @warn("Only model parameters are saved but no information to reconstruct the model.")
      end
   else 
      D["model_spec"] = model_spec
   end

   if !isnothing(errors)
      D["errors"] = errors
   end

   open(filename, "w") do io
       write(io, JSON.json(D, 3))
   end

   if verbose
      @info "Results saved to file: $filename"
   end
end


function load_model(filename) 
   D = JSON.parsefile(filename)
   model = make_model(D["model_spec"])
   set_parameters!(model, D["model_parameters"])
   return model, D
end