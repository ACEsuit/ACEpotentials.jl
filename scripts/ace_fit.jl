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
    # doesn't work properly somehow
    data = ACE1pack.read_data(d)    
    global n_obs = 0
    for (okey, d, _) in IPFitting.observations(data)
        len = length(IPFitting.observation(d, okey))
        n_obs += len
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
nprocs = args["num-blas-threads"]
if nprocs > 1
    using LinearAlgebra
    @warn "Using $nprocs threads for BLAS"
    BLAS.set_num_threads(nprocs)
end

raw_params = load_dict(args["params"])
fit_params = fill_defaults(raw_params)

if args["dry-run"]
    save_dry_run_info(fit_params)
end

ACE1pack.fit_ace(fit_params)
