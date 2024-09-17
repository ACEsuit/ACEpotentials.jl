#
# See ACEpotentials.jl documentation for correct usage of this script. 
#

using ACEpotentials
using ACEpotentials: JSON, ExtXYZ
using TOML
using ACEpotentials.ArgParse: ArgParseSettings, @add_arg_table!, parse_args

parser = ArgParseSettings(description="Fit an ACE potential from parameters file")

@add_arg_table! parser begin
    "--params", "-p"
        help = "A JSON or YAML filename with parameters for the fit"
    "--dry-run"
        help = "Only quickly compute various sizes, etc"
        action = :store_true
    "--num-blas-threads"
        help = "Number of processes for BLAS to use when solving the LsqDB"
        arg_type = Int
        default = 1
    "--result_folder", "-o"
        help = "folder path to store all results"    
end

# parse the command 
args = parse_args(parser)

@info("Load parameter file") 
args_dict = JSON.parsefile(args["params"])

# outputs
if args["result_folder"] === nothing
    @info("result_folder not specified, create result folder at $(args["params"][1:end-5] * "results")")
    args["result_folder"] = args["params"][1:end-5] * "_results"
end
res_path = args["result_folder"]
mkpath(res_path)
@info("result storing at $(res_path)")

@info("Construct ACEmodel of type $(args_dict["model"]["model_name"])")
model = ACEpotentials.make_model(args_dict["model"])

@info("Load datasets")  
train = ExtXYZ.load(args_dict["data"]["train_file"])

data_keys = (
    force_key = args_dict["data"]["force_key"],
    energy_key = args_dict["data"]["energy_key"],
    virial_key = args_dict["data"]["virial_key"])

weights = args_dict["solve"]["weights"]

solver = ACEpotentials.make_solver(model, 
                                   args_dict["solve"]["solver"], 
                                   args_dict["solve"]["prior"])
 
acefit!(train, model;
       data_keys..., 
       weights = weights, 
       solver = solver)

# --- saving results and model below ---
D = Dict()
OD = args_dict["output"]

# train/test errors
if OD["error_table"] || OD["scatter"]
    @info("evaluating errors")
    # training errors
    err_train, train_evf = ACEpotentials.compute_errors(train, model; data_keys..., weights=weights, return_efv = true)
    err = Dict("train" => err_train)
    if OD["scatter"]
        D["train_evf"] = train_evf
    end

    # test errors (if a test dataset exists)
    if haskey(args_dict["data"], "test_file")
        test = ExtXYZ.load(args_dict["data"]["test_file"])
        err_test, test_evf = ACEpotentials.compute_errors(test, model; data_keys..., weights=weights, return_efv = true)
        err["test"] = err_test
        if OD["scatter"]
            D["test_evf"] = test_evf
        end
    end
end

# dimer analysis if specified
if args_dict["output"]["dimer"]
    D["dimers"] = ACEpotentials.dimers(model, args_dict["model"]["elements"])
end

# saving results to folder
ACEpotentials.save_model(model, joinpath(res_path, args_dict["output"]["model"]),
                         model_spec = args_dict, 
                         errors = err,
                         save_project = args_dict["output"]["save_project"],
                         meta = D)

# To load the model, active the same Julia environment, then run 
# `model, meta = ACEpotentials.load_model("path/to/model.json")`
# the resulting `model` object should be equivalent to the fitted `model`.

if args_dict["output"]["error_table"]
    et_file = open(joinpath(res_path, "error_table.txt"), "w")
    ori_stdout = stdout; ori_stderr = stderr
    redirect_stdio(stdout=et_file, stderr = et_file)
    @info("Training error")
    print_errors_tables(err_train)
    @info("Testing error")
    print_errors_tables(err_test)
    redirect_stdio(stdout=ori_stdout, stderr=ori_stderr); 
    close(et_file)
end

# --- make plots if specified ---
if args_dict["output"]["make_plots"]
    # 1. scatter EFV
    if args_dict["output"]["scatter"]
        using Plots
        function scatter_quantity(Xs, Ys; args...)
            p = scatter(Xs, Ys, markersize=5; args...)
            min_val = min(minimum(Xs), minimum(Ys))
            max_val = max(maximum(Xs), maximum(Ys))
            plot!([min_val, max_val], [min_val, max_val], linestyle=:dash, color=:red, linewidth=2.0)
            return p
        end
        function concat_preallocate(vectors)
            total_length = sum(length(v) for v in vectors)
            result = zeros(Float64, total_length)
            pos = 1
            for v in vectors
                result[pos:pos+length(v)-1] .= v
                pos += length(v)
            end
            return result
        end
        PP = Dict("train" => Dict(), "test" => Dict())
        for X in ("E", "F", "V")
            PP["train"][X] = scatter_quantity(concat_preallocate(train_evf[X * "pred"]),
                               concat_preallocate(train_evf[X * "ref"]);
                               title = X * "train", xlabel = "predicted", markerstrokewidth=0,
                               ylabel = "ground truth", legend = nothing,
                               alpha = 0.5
                               )
            print("Done train")
            PP["test"][X] = scatter_quantity(concat_preallocate(test_evf[X * "pred"]),
                                concat_preallocate(test_evf[X * "ref"]);
                               title = X * "test", xlabel = "predicted", markerstrokewidth=0,
                               ylabel = "ground truth", legend = nothing,
                               alpha = 0.5
                               )
        end
        pall = plot(
            PP["train"]["E"], PP["test"]["E"], 
            PP["train"]["F"], PP["test"]["F"], 
            PP["test"]["V"], PP["test"]["V"],
            layout=(3, 2), size=(800, 800),
        )
        savefig(pall, joinpath(res_path, "scatter.png"))
    end

    # 2. dimer analysis
    if args_dict["output"]["dimer"]
        ZZ = args_dict["model"]["elements"]
        for i = 1:length(args_dict["model"]["elements"]), j = 1:i
            Dij = D["dimers"][(ZZ[i], ZZ[j])]
            # take > -10 eV portions
            idx = ustrip.(Dij[2]) .> -10
            p = plot(Dij[1][idx], -Dij[2][idx], xlabel = "r", ylabel = "E")
            savefig(p, joinpath(res_path, "dimer_[$(ZZ[i]), $(ZZ[j])].png"))
        end
    end
end
