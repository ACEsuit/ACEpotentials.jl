using Test
using ACE1pack

params = load_dict("Si_tiny_params.yaml")
params = fill_defaults!(params)
include("Si_tiny_rmse.jl")

function test_rmse(rmse, expected, atol)
    for config in keys(rmse)
        for obs in keys(rmse[config])
            @test rmse[config][obs] ≈ expected[config][obs] atol=atol
        end
    end
end

@testset "QR" begin
    params["solver"]["type"] = "qr" 
    coef, fit_info = fit_ace(params)
    test_rmse(fit_info["errors"]["rmse"], rmse_qr, 1e-5)
end

@testset "LSQR" begin
    params["solver"]["type"] = "lsqr"
    params["solver"]["lsqr_damp"] = 2e-2
    params["solver"]["lsqr_conlim"] = 1e12
    params["solver"]["lsqr_atol"] = 1e-7
    params["solver"]["lsqr_maxiter"] = 100000
    coef, fit_info = fit_ace(params)
    test_rmse(fit_info["errors"]["rmse"], rmse_lsqr, 1e-5)
end

@testset "RRQR" begin
    params["solver"]["type"] = "rrqr" 
    params["solver"]["rrqr_tol"] = 1e-12
    coef, fit_info = fit_ace(params)
    test_rmse(fit_info["errors"]["rmse"], rmse_rrqr, 1e-5)
end

@testset "BRR" begin
    params["solver"]["type"] = "brr"
    params["solver"]["brr_tol"] = 1e-4
    coef, fit_info = fit_ace(params)
    test_rmse(fit_info["errors"]["rmse"], rmse_brr, 1e-5)
end

@testset "ARD" begin
    params["solver"]["type"] = "ard"
    params["solver"]["ard_tol"] = 2e-3
    params["solver"]["ard_threshold_lambda"] = 5000
    params["solver"]["ard_n_iter"] = 1000
    coef, fit_info = fit_ace(params)
    test_rmse(fit_info["errors"]["rmse"], rmse_ard, 1e-5)
end
