using Pkg
Pkg.activate(joinpath(@__DIR__(), "../"))

using ACEpotentials, AtomsBase, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Zygote, Optimisers, Folds, Printf, Optim, Random, JSON, ArgParse

M = ACEpotentials.Models
rng = Random.GLOBAL_RNG

include("../docs/src/newkernels/llsq.jl") 

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

# TODO: make it work later, mt assemble is not working with the new kernel yet
# nprocs = args["num-blas-threads"]
# if nprocs > 1
#     using LinearAlgebra
#     @info "Using $nprocs threads for BLAS"
#     BLAS.set_num_threads(nprocs)
#     controller = pyimport("threadpoolctl")["ThreadpoolController"]()
#     controller.limit(limits=nprocs, user_api="blas")
#     pyimport("pprint")["pprint"](controller.select(user_api="blas").info())
# end

@info("making ACEmodel")
model = ACEpotentials.make_acemodel(args_dict["model"])

# convet the model into lux calculator
ps, st = Lux.setup(rng, model)
calc_model = M.ACEPotential(model, ps, st)

# load data from example dataset, in practice it should be 
# ```
# train = read_extxyz(args_dict["data"]["in_data"]["train_file"])
# test = read_extxyz(args_dict["data"]["in_data"]["test_file"])
# ```
train, test, _ = ACEpotentials.example_dataset(args_dict["data"]["in_data"]["train_file"])
train = train[1:3:end]

# wrap this into AtomsBase format
train2 = FlexibleSystem.(train)
test2 = FlexibleSystem.(test)

# things that we should get from dict
data_keys = (E_key = :energy, F_key = :force, )

# TODO: in here weight should be also depending on config_type
wE = 30.0; wF = 1.0; wV = 1.0
weights = (wE = wE/u"eV", wF = wF / u"eV/Å", )
A, y = LLSQ.assemble_lsq(calc_model, train2, weights, data_keys)
θ = ACEfit.trunc_svd(svd(A), y, 1e-8)
calc_model2_fit = LLSQ.set_linear_params(calc_model, θ)

##

# errors
E_train, F_train = LLSQ.rmse(train2, calc_model2_fit)
E_test, F_test = LLSQ.rmse(test2, calc_model2_fit)

@printf("       |      E    |    F  \n")
@printf(" train | %.2e  |  %.2e  \n", E_train, F_train)
@printf("  test | %.2e  |  %.2e  \n", E_test, F_test)