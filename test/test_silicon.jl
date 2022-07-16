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
        for obs in keys(rmse[config])
            @test rmse[config][obs] ≈ expected[config][obs] atol=atol
        end
    end
end

@testset "QR" begin
    params["solver"]["type"] = "qr" 
    rmse_qr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0234649, "E"=>0.000617953, "F"=>0.018611),
        "liq"           => Dict("V"=>0.034633, "E"=>0.000133371, "F"=>0.104112),
        "set"           => Dict("V"=>0.0437043, "E"=>0.00128242, "F"=>0.0819438),
        "bt"            => Dict("V"=>0.0576748, "E"=>0.0017616, "F"=>0.0515637),)
    IP, fit_info = fit_ace(params)
    test_rmse(fit_info["errors"]["rmse"], rmse_qr, 1e-5)
end

@testset "LSQR" begin
    params["solver"]["type"] = "lsqr"
    params["solver"]["lsqr_damp"] = 2e-2
    params["solver"]["lsqr_conlim"] = 1e12
    params["solver"]["lsqr_atol"] = 1e-7
    params["solver"]["lsqr_maxiter"] = 100000
    rmse_lsqr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0414375, "E"=>0.00179828, "F"=>0.0308943),
        "liq"           => Dict("V"=>0.0340089, "E"=>0.000770027, "F"=>0.1795),
        "set"           => Dict("V"=>0.0687971, "E"=>0.00276312, "F"=>0.138958),
        "bt"            => Dict("V"=>0.0896389, "E"=>0.00359229, "F"=>0.0706966),)
    IP, fit_info = fit_ace(params)
    @warn "The LSQR test tolerance is very loose."
    test_rmse(fit_info["errors"]["rmse"], rmse_lsqr, 1e-1)
end

@testset "RRQR" begin
    params["solver"]["type"] = "rrqr" 
    params["solver"]["rrqr_tol"] = 1e-12
    rmse_rrqr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0234649, "E"=>0.000617953, "F"=>0.018611),
        "liq"           => Dict("V"=>0.034633, "E"=>0.000133371, "F"=>0.104112),
        "set"           => Dict("V"=>0.0437043, "E"=>0.00128242, "F"=>0.0819438),
        "bt"            => Dict("V"=>0.0576748, "E"=>0.0017616, "F"=>0.0515637),)
    IP, fit_info = fit_ace(params)
    test_rmse(fit_info["errors"]["rmse"], rmse_rrqr, 1e-5)
end

@testset "BRR" begin
    params["solver"]["type"] = "brr"
    params["solver"]["brr_tol"] = 1e-4
    rmse_brr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0428196, "E"=>0.00154743, "F"=>0.0286633),
        "liq"           => Dict("V"=>0.0346353, "E"=>9.80216e-5, "F"=>0.152752),
        "set"           => Dict("V"=>0.0664363, "E"=>0.00240266, "F"=>0.11819),
        "bt"            => Dict("V"=>0.0851538, "E"=>0.00313734, "F"=>0.0585017),)
    IP, fit_info = fit_ace(params)
    test_rmse(fit_info["errors"]["rmse"], rmse_brr, 1e-5)
end

@testset "ARD" begin
    params["solver"]["type"] = "ard"
    params["solver"]["ard_tol"] = 2e-3
    params["solver"]["ard_threshold_lambda"] = 5000
    params["solver"]["ard_n_iter"] = 1000
    rmse_ard = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0477465, "E"=>0.00156758, "F"=>0.0417769),
        "liq"           => Dict("V"=>0.0367959, "E"=>0.0008153, "F"=>0.176037),
        "set"           => Dict("V"=>0.0734983, "E"=>0.00302168, "F"=>0.136656),
        "bt"            => Dict("V"=>0.0940645, "E"=>0.00410442, "F"=>0.0667358),)
    IP, fit_info = fit_ace(params)
    @warn "The ARD test tolerance is very loose."
    test_rmse(fit_info["errors"]["rmse"], rmse_ard, 1e-2)
end
