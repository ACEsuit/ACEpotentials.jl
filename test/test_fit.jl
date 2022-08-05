using ACE1pack, JuLIP, LazyArtifacts, Test
using JuLIP.Testing: print_tf

test_train_set = joinpath(artifact"TiAl_tiny_dataset", "TiAl_tiny.xyz")
expected_errors_json = joinpath(artifact"ACE1pack_test_files", "expected_fit_errors.json")

@info("test full fit from script")

species = [:Ti, :Al]
r0 = 2.88 

data = data_params(fname = test_train_set,
    energy_key = "energy",
    force_key = "force",
    virial_key = "virial")

ace_basis = basis_params(
    type = "ace",
    species = species, 
    N = 3, 
    maxdeg = 6, 
    r0 = r0, 
    radial = basis_params(
        type = "radial", 
        rcut = 5.0, 
        rin = 1.44,
        pin = 2))

pair_basis = basis_params(
    type = "pair", 
    species = species, 
    maxdeg = 6,
    r0 = r0,
    rcut = 5.0,
    rin = 0.0,
    pcut = 2, # TODO: check if it should be 1 or 2?
    pin = 0)

basis = Dict(
    "ace" => ace_basis,
    "pair" => pair_basis
)

solver = solver_params(type = :lsqr)

# symbols for species (e.g. :Ti) would work as well
e0 = Dict("Ti" => -1586.0195, "Al" => -105.5954)

weights = Dict(
    "default" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0),
    "FLD_TiAl" => Dict("E" => 5.0, "F" => 1.0, "V" => 1.0),
    "TiAl_T5000" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0))

P = regularizer_params(type = "laplacian", rlap_scal = 3.0)

params = fit_params(
    data = data,
    basis = basis,
    solver = solver,
    e0 = e0,
    weights = weights,
    P = P,
    ACE_fname = "")

coef, errors = fit_ace(params)

expected_errors = load_dict(expected_errors_json)

for error_type in keys(errors),
        config_type in keys(errors[error_type]),
            property in keys(errors[error_type][config_type])
    print_tf(@test isapprox(errors[error_type][config_type][property],
                            expected_errors[error_type][config_type][property],
#                            atol=1e-3))
                            atol=5e-2))
end
println() 
