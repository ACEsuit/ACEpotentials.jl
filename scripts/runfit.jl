using Pkg
Pkg.activate(joinpath(@__DIR__(), "../"))

using ACEpotentials, AtomsBase, AtomsBuilder, Lux, StaticArrays, LinearAlgebra, 
      Unitful, Zygote, Optimisers, Folds, Printf, Optim, Random, JSON, ArgParse


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

#get_basis_size(d::Dict) =
#    sum([length(ACEpotentials.generate_basis(basis_params)) for (basis_name, basis_params) in d])

function get_num_observations(d::Dict)

    data = JuLIP.read_extxyz(d["fname"])
    global n_obs = 0
    for atoms in data
        n_obs += length(atoms.Z)  # Z - atomic numbers
    end
    return n_obs
end

# parse arg
args = parse_args(parser)

# load from json
args_dict = load_dict(joinpath(@__DIR__(), "fitting_params.json"))

# pretty print
print(json(args_dict, 3))

# TODO: chho: make it work later
nprocs = args["num-blas-threads"]
if nprocs > 1
    using LinearAlgebra
    @info "Using $nprocs threads for BLAS"
    BLAS.set_num_threads(nprocs)
    controller = pyimport("threadpoolctl")["ThreadpoolController"]()
    controller.limit(limits=nprocs, user_api="blas")
    pyimport("pprint")["pprint"](controller.select(user_api="blas").info())
end

include("../src/ace1_compat.jl")
include("../scripts/json_interface.jl")


@info("making ACEmodel")
model = make_acemodel(args_dict["model"])

# convet the model into lux calculator
rng = Random.GLOBAL_RNG
ps, st = Lux.setup(rng, model)
M = ACEpotentials.Models
calc_model = M.ACEPotential(model, ps, st)

# linear system assmeble with the lux calculator
Z0 = :Si
train, test, _ = ACEpotentials.example_dataset("Zuo20_$Z0")
train = train[1:3:end]
wE = 30.0; wF = 1.0; wV = 1.0

train2 = FlexibleSystem.(train)
test2 = FlexibleSystem.(test)

data_keys = (E_key = :energy, F_key = :force, )
weights = (wE = wE/u"eV", wF = wF / u"eV/Å", )

A, y = LLSQ.assemble_lsq(calc_model, train2, weights, data_keys)
@show size(A)

θ = ACEfit.trunc_svd(svd(A), y, 1e-8)

calc_model2_fit = LLSQ.set_linear_params(calc_model, θ)


# errors

E_train, F_train = LLSQ.rmse(train2, calc_model2_fit)
E_test, F_test = LLSQ.rmse(test2, calc_model2_fit)

@printf("       |      E    |    F  \n")
@printf(" train | %.2e  |  %.2e  \n", E_train, F_train)
@printf("  test | %.2e  |  %.2e  \n", E_test, F_test)