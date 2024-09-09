using Pkg
Pkg.activate(joinpath(@__DIR__(), "../"))

using ACEpotentials, AtomsBase, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Zygote, Optimisers, Folds, Printf, Optim, Random, JSON, ArgParse, Dates, ExtXYZ

M = ACEpotentials.Models
rng = Random.GLOBAL_RNG

parser = ArgParseSettings(description="Fit an ACE potential from parameters file")
@add_arg_table parser begin
    "--params", "-p"
        help = "A JSON or YAML filename with parameters for the fit"
    "--dry-run"
        help = "Only quickly compute various sizes, etc"
        action = :store_true
    "--num-blas-threads"
        help = "Number of processes for BLAS to use when solving the LsqDB"
        arg_type = Int
        default = 1
end

# parse arg
args = parse_args(parser)

# load from json
args_dict = load_dict(args["params"])

@info("making ACEmodel")
calc_model = let model = ACEpotentials.make_acemodel(args_dict["model"]); ps, st = Lux.setup(rng, model); M.ACEPotential(model, ps, st); end

# Load the training data 
train = ExtXYZ.load(args_dict["data"]["in_data"]["train_file"])
test = ExtXYZ.load(args_dict["data"]["in_data"]["test_file"])

data_keys = Dict(:energy_key => "dft_energy",
                 :force_key => "dft_force",
                 :virial_key => "dft_virial")
weights = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0),
               "liq" => Dict("E"=>10.0, "F"=>0.66, "V"=>0.25))
        
acefit!(calc_model, train;
        data_keys...,
        weights = weights,
        solver=ACEfit.LSQR())

# errors
err_train = ACEpotentials.linear_errors(train, calc_model; data_keys..., weights=weights)
err_test = ACEpotentials.linear_errors(test, calc_model; data_keys..., weights=weights)
err = Dict("train" => err_train, "test" => err_test)

# save results
function nested_namedtuple_to_dict(nt)
    return Dict(k => isa(v, NamedTuple) ? nested_namedtuple_to_dict(v) : v for (k, v) in pairs(nt))
end

function save_results_to_file(args_dict, err, calc_model, filename)
    model_params_dict = nested_namedtuple_to_dict(calc_model.ps)
    results = Dict(
        "args_dict" => args_dict,
        "err" => err,
        "model_parameters" => model_params_dict
    )
    open(filename, "w") do io
        write(io, JSON.json(results, 4))
    end
    @info "Results saved to file: $filename"
end

save_results_to_file(args_dict, err, calc_model, "scripts/results.json")
