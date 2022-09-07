using Distributed
using Test
using LazyArtifacts
using ACE1pack

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

@testset "QR" begin
    params["solver"] = Dict{Any,Any}("type" => "qr")
    rmse_qr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0234649, "E"=>0.000617953, "F"=>0.018611),
        "liq"           => Dict("V"=>0.034633, "E"=>0.000133371, "F"=>0.104112),
        "set"           => Dict("V"=>0.0437043, "E"=>0.00128242, "F"=>0.0819438),
        "bt"            => Dict("V"=>0.0576748, "E"=>0.0017616, "F"=>0.0515637),)
    IP, errors = fit_ace(params)
    test_rmse(errors["rmse"], rmse_qr, 1e-5)

    # repeat with distributed assembly
    addprocs(2)
    @everywhere using ACE1pack
    IP, errors = fit_ace(params, :distributed)
    rmprocs(workers())
    test_rmse(errors["rmse"], rmse_qr, 1e-5)
end

@testset "LSQR" begin
    params["solver"] = Dict{Any,Any}("type" => "lsqr")
    params["solver"]["damp"] = 2e-2
    params["solver"]["conlim"] = 1e12
    params["solver"]["atol"] = 1e-7
    params["solver"]["maxiter"] = 100000
    params["solver"]["verbose"] = false
    rmse_lsqr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0414375, "E"=>0.00179828, "F"=>0.0308943),
        "liq"           => Dict("V"=>0.0340089, "E"=>0.000770027, "F"=>0.1795),
        "set"           => Dict("V"=>0.0687971, "E"=>0.00276312, "F"=>0.138958),
        "bt"            => Dict("V"=>0.0896389, "E"=>0.00359229, "F"=>0.0706966),)
    IP, errors = fit_ace(params)
    @warn "The LSQR test tolerance is very loose."
    test_rmse(errors["rmse"], rmse_lsqr, 1e-1)
end

@testset "RRQR" begin
    params["solver"] = Dict{Any,Any}("type" => "rrqr")
    params["solver"]["rtol"] = 1e-12
    rmse_rrqr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0234649, "E"=>0.000617953, "F"=>0.018611),
        "liq"           => Dict("V"=>0.034633, "E"=>0.000133371, "F"=>0.104112),
        "set"           => Dict("V"=>0.0437043, "E"=>0.00128242, "F"=>0.0819438),
        "bt"            => Dict("V"=>0.0576748, "E"=>0.0017616, "F"=>0.0515637),)
    IP, errors = fit_ace(params)
    test_rmse(errors["rmse"], rmse_rrqr, 1e-5)
end

@testset "SKLEARN_BRR" begin
    params["solver"] = Dict{Any,Any}("type" => "sklearn_brr")
    params["solver"]["tol"] = 1e-4
    rmse_brr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0428196, "E"=>0.00154743, "F"=>0.0286633),
        "liq"           => Dict("V"=>0.0346353, "E"=>9.80216e-5, "F"=>0.152752),
        "set"           => Dict("V"=>0.0664363, "E"=>0.00240266, "F"=>0.11819),
        "bt"            => Dict("V"=>0.0851538, "E"=>0.00313734, "F"=>0.0585017),)
    IP, errors = fit_ace(params)
    test_rmse(errors["rmse"], rmse_brr, 1e-5)
end

@testset "SKLEARN_ARD" begin
    params["solver"] = Dict{Any,Any}("type" => "sklearn_ard")
    params["solver"]["tol"] = 2e-3
    params["solver"]["threshold_lambda"] = 5000
    params["solver"]["n_iter"] = 1000
    rmse_ard = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0477465, "E"=>0.00156758, "F"=>0.0417769),
        "liq"           => Dict("V"=>0.0367959, "E"=>0.0008153, "F"=>0.176037),
        "set"           => Dict("V"=>0.0734983, "E"=>0.00302168, "F"=>0.136656),
        "bt"            => Dict("V"=>0.0940645, "E"=>0.00410442, "F"=>0.0667358),)
    IP, errors = fit_ace(params)
    @warn "The SKLEARN_ARD test tolerance is very loose."
    test_rmse(errors["rmse"], rmse_ard, 1e-2)
end

@testset "BLR" begin
    params["solver"] = Dict{Any,Any}("type" => "BLR")
    rmse_blr = Dict(
         "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
         "set"           => Dict("V"=>0.0619346, "E"=>0.00238807, "F"=>0.121907),
         "dia"           => Dict("V"=>0.0333255, "E"=>0.00130242, "F"=>0.0255582),
         "liq"           => Dict("V"=>0.0345897, "E"=>0.000397724, "F"=>0.157461),
         "bt"            => Dict("V"=>0.0822944, "E"=>0.00322198, "F"=>0.062758),)
    IP, errors = fit_ace(params)
    test_rmse(errors["rmse"], rmse_blr, 1e-5)
end


