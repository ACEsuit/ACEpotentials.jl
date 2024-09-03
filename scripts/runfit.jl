using Pkg
Pkg.activate(joinpath(@__DIR__(), "../"))

using ACEpotentials, AtomsBase, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Zygote, Optimisers, Folds, Printf, Optim, Random, JSON, ArgParse, ExtXYZ, Dates

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
model = ACEpotentials.make_acemodel(args_dict["model"])

# convet the model into lux calculator
ps, st = Lux.setup(rng, model)
calc_model = M.ACEPotential(model, ps, st)

train = ExtXYZ.load(args_dict["data"]["in_data"]["train_file"])
test = ExtXYZ.load(args_dict["data"]["in_data"]["test_file"])

# wrap this into AtomsBase format
# train2 = FlexibleSystem.(train)
# test2 = FlexibleSystem.(test)

data_keys = [:energy_key => "dft_energy",
             :force_key => "dft_force",
             :virial_key => "dft_virial"]
weights = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0),
               "liq" => Dict("E"=>10.0, "F"=>0.66, "V"=>0.25))
        
acefit!(calc_model, train;
        data_keys...,
        weights = weights,
        solver=ACEfit.LSQR())

# errors
E_train, F_train = ACEpotentials.linear_errors(train, calc_model; data_keys..., weights=weights)
E_test, F_test = ACEpotentials.linear_errors(test, calc_model; data_keys..., weights=weights)


function save_results_to_disk(E_train, F_train, E_test, F_test; dir="results", filename="results_log.txt")
    mkpath(dir)
    content = """
    Files created: $(now()) 
    ----------------------------------------
           |      E    |    F  
     train | $(E_train)  |  $(F_train)  
      test | $(E_test)  |  $(F_test)  
    """
    open(joinpath(dir, filename), "w") do file
        write(file, content)
    end
end

save_results_to_disk(E_train, F_train, E_test, F_test)
