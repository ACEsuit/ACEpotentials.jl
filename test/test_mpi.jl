# With JULIA_PROJECT set to ACE1pack base directory.

using ACE1pack
using LazyArtifacts
using Test

include("../scripts/ACE1packMPI.jl")

### ----- set up params -----

data = data_params(
    fname=joinpath(artifact"Si_tiny_dataset", "Si_tiny.xyz"),
    energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
ace_basis = basis_params(
    type="ace",
    species=[:Si],
    N=4,
    maxdeg=12,
    r0=2.35126,
    radial=basis_params(type="radial", pin=2, pcut=2, rcut=5.5, rin=1.65))
pair_basis = basis_params(
    type="pair",
    species=[:Si],
    maxdeg=3,
    r0=2.35126,
    rcut=6.5,
    pcut=1,
    pin=0)
basis = Dict("ace"=>ace_basis, "pair"=>pair_basis)
solver = solver_params(type=:qr)
e0 = Dict("Si"=>-158.54496821)
weights = Dict(
    "default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0),
    "liq" => Dict("E"=>10.0, "F"=>0.66, "V"=>0.25))
params = fit_params(
    data = data,
    basis = basis,
    solver = solver,
    e0 = e0,
    weights = weights,
    ACE_fname = "")

### ----- perform tests -----

function test_rmse(rmse, expected, atol)
    for config in keys(rmse)
        # TODO and/or warning: can't iterate over rmse because it will have virial for isolated atom
        #for obs in keys(rmse[config])
        for obs in keys(expected[config])
            @test rmse[config][obs] ≈ expected[config][obs] atol=atol
        end
    end
end

@testset "LeastSquares" begin
    params["solver"] = Dict{Any,Any}("type" => "qr")
    rmse_qr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0234649, "E"=>0.000617953, "F"=>0.018611),
        "liq"           => Dict("V"=>0.034633, "E"=>0.000133371, "F"=>0.104112),
        "set"           => Dict("V"=>0.0437043, "E"=>0.00128242, "F"=>0.0819438),
        "bt"            => Dict("V"=>0.0576748, "E"=>0.0017616, "F"=>0.0515637),)
    IP, errors = ace_fit_mpi(params)
    test_rmse(errors["rmse"], rmse_qr, 1e-5)
end

ACE1packMPI_finalize()
