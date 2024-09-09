#
# julia --project=. runfit.jl -p example_params.json
#

using ACEpotentials
using ACEpotentials: JSON, ExtXYZ
using ACEpotentials.ArgParse: ArgParseSettings, @add_arg_table, parse_args

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

# parse the command 
args = parse_args(parser)

@info("Load parameter file") 
args_dict = JSON.parsefile(args["params"])

@info("Construct ACEmodel of type $(args_dict["model"]["model_name"])")
model = ACEpotentials.make_model(args_dict["model"])

@info("Load datasets")  
train = ExtXYZ.load(args_dict["data"]["train_file"])

data_keys = (
    force_key = args_dict["data"]["force_key"],
    energy_key = args_dict["data"]["energy_key"],
    virial_key = args_dict["data"]["virial_key"])

weights = args_dict["solve"]["weights"]

solver = ACEpotentials.make_solver(args_dict["solve"]["solver"])
 
# TODO: make prior

acefit!(train, model;
       data_keys..., weights = weights, solver = solver)

# training errors
err_train = ACEpotentials.linear_errors(train, model; data_keys..., weights=weights)
err = Dict("train" => err_train)

# test errors (if a test dataset exists)
if haskey(args_dict["data"], "test_file")
    test = ExtXYZ.load(args_dict["data"]["test_file"])
    err_test = ACEpotentials.linear_errors(test, model; data_keys..., weights=weights)
    err["test"] = err_test
end

# saving results
result_file = args_dict["output"]["model"]
ACEpotentials.save_model(model, @__DIR__() * "/results.json"; 
                         make_model_args = args_dict, 
                         errors = err, )
