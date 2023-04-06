using ACE1pack

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

get_basis_size(d::Dict) = 
    sum([length(ACE1pack.generate_basis(basis_params)) for (basis_name, basis_params) in d])

function get_num_observations(d::Dict)

    data = JuLIP.read_extxyz(d["fname"])
    global n_obs = 0
    for atoms in data
        n_obs += length(atoms.Z)  # Z - atomic numbers
    end
    return n_obs
end

function save_dry_run_info(fit_params)
    num_observations = get_num_observations(fit_params["data"])
    basis_size = get_basis_size(fit_params["basis"])
    dry_fit_filename = replace(fit_params["ACE_fname"], ".json" => ".size")
    size_info = Dict("lsq_matrix_shape" => (num_observations, basis_size))
    save_json(dry_fit_filename, size_info)
    exit(0)
end

args = parse_args(parser)
raw_params = load_dict(args["params"])
fit_params = fill_defaults(raw_params)

# the export to lammps try to replace .json with .yace, so check that ACE_fname actually ends in .json first
if fit_params["ACE_fname"][end-4:end] != ".json"
    throw("potential file names must end in .json")
end

nprocs = args["num-blas-threads"]
if nprocs > 1
    using LinearAlgebra
    @info "Using $nprocs threads for BLAS"
    BLAS.set_num_threads(nprocs)
    controller = pyimport("threadpoolctl")["ThreadpoolController"]()
    controller.limit(limits=nprocs, user_api="blas")
    pyimport("pprint")["pprint"](controller.select(user_api="blas").info())
end

if args["dry-run"]
    save_dry_run_info(fit_params)
end

results = ACE1pack.fit_ace(fit_params)

# export to a .yace automatically, also need to generate the new name.
yace_name = replace(fit_params["ACE_fname"], ".json" => ".yace")
ACE1pack.ExportMulti.export_ACE(yace_name, results["IP"]; export_pairpot_as_table=true)
