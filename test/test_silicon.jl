using ACEpotentials
using Distributed
using LazyArtifacts
using PythonCall
using Test

## ----- setup -----

@warn "test_silicon not fully converted yet."
model = acemodel(elements = [:Si],
                 Eref = [:Si => -158.54496821],
                 rcut = 5.5,
                 order = 3,
                 totaldegree = 12)
data = read_extxyz(artifact"Si_tiny_dataset" * "/Si_tiny.xyz")
data_keys = [:energy_key => "dft_energy",
             :force_key => "dft_force",
             :virial_key => "dft_virial"]
weights = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0),
               "liq" => Dict("E"=>10.0, "F"=>0.66, "V"=>0.25))

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
    rmse_qr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0234649, "E"=>0.000617953, "F"=>0.018611),
        "liq"           => Dict("V"=>0.034633, "E"=>0.000133371, "F"=>0.104112),
        "set"           => Dict("V"=>0.0437043, "E"=>0.00128242, "F"=>0.0819438),
        "bt"            => Dict("V"=>0.0576748, "E"=>0.0017616, "F"=>0.0515637),)
    acefit!(model, data;
            data_keys...,
            weights = weights,
            solver=ACEfit.QR())
    #test_rmse(results["errors"]["rmse"], rmse_qr, 1e-5)

    # repeat with distributed assembly
    addprocs(3, exeflags="--project=$(Base.active_project())")
    @everywhere using ACEpotentials
    acefit!(model, data;
            data_keys...,
            weights = weights,
            solver=ACEfit.QR())
    rmprocs(workers())
    #test_rmse(results["errors"]["rmse"], rmse_qr, 1e-5)
end

@testset "LSQR" begin
    rmse_lsqr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0414375, "E"=>0.00179828, "F"=>0.0308943),
        "liq"           => Dict("V"=>0.0340089, "E"=>0.000770027, "F"=>0.1795),
        "set"           => Dict("V"=>0.0687971, "E"=>0.00276312, "F"=>0.138958),
        "bt"            => Dict("V"=>0.0896389, "E"=>0.00359229, "F"=>0.0706966),)
    acefit!(model, data;
            data_keys...,
            weights = weights,
            solver = ACEfit.LSQR(
                damp = 2e-2, conlim = 1e12, atol = 1e-7,
                maxiter = 100000, verbose = false))
    @warn "The LSQR test tolerance is very loose."
    #test_rmse(results["errors"]["rmse"], rmse_lsqr, 1e-1)
end

@testset "RRQR" begin
    rmse_rrqr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0234649, "E"=>0.000617953, "F"=>0.018611),
        "liq"           => Dict("V"=>0.034633, "E"=>0.000133371, "F"=>0.104112),
        "set"           => Dict("V"=>0.0437043, "E"=>0.00128242, "F"=>0.0819438),
        "bt"            => Dict("V"=>0.0576748, "E"=>0.0017616, "F"=>0.0515637),)
    acefit!(model, data;
            data_keys...,
            weights = weights,
            solver = ACEfit.RRQR(rtol = 1e-12))
    #test_rmse(results["errors"]["rmse"], rmse_rrqr, 1e-5)
end

@testset "SKLEARN_BRR" begin
    rmse_brr = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.0333241, "E"=>0.0013034, "F"=>0.0255757),
        "liq"           => Dict("V"=>0.0347208, "E"=>0.0003974, "F"=>0.1574544),
        "set"           => Dict("V"=>0.0619434, "E"=>0.0023868, "F"=>0.1219008),
        "bt"            => Dict("V"=>0.0823042, "E"=>0.0032196, "F"=>0.0627417),)
    acefit!(model, data;
            data_keys...,
            weights = weights,
            solver = ACEfit.SKLEARN_BRR(tol = 1e-4))
    #test_rmse(results["errors"]["rmse"], rmse_brr, 1e-5)
end

@testset "SKLEARN_ARD" begin
    rmse_ard = Dict(
        "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
        "dia"           => Dict("V"=>0.1084975, "E"=>0.0070814, "F"=>0.0937790),
        "liq"           => Dict("V"=>0.0682268, "E"=>0.0090065, "F"=>0.3693146),
        "set"           => Dict("V"=>0.1839696, "E"=>0.0137778, "F"=>0.2883043),
        "bt"            => Dict("V"=>0.2413568, "E"=>0.0185958, "F"=>0.1507498),)
    acefit!(model, data;
            data_keys...,
            weights = weights,
            solver = ACEfit.SKLEARN_ARD(
                tol = 2e-3, threshold_lambda = 5000, n_iter = 1000))
    @warn "The SKLEARN_ARD test tolerance is very loose."
    #test_rmse(results["errors"]["rmse"], rmse_ard, 1e-2)
end

@testset "BLR" begin
    rmse_blr = Dict(
         "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
         "set"           => Dict("V"=>0.0619346, "E"=>0.00238807, "F"=>0.121907),
         "dia"           => Dict("V"=>0.0333255, "E"=>0.00130242, "F"=>0.0255582),
         "liq"           => Dict("V"=>0.0345897, "E"=>0.000397724, "F"=>0.157461),
         "bt"            => Dict("V"=>0.0822944, "E"=>0.00322198, "F"=>0.062758),)
    acefit!(model, data;
            data_keys...,
            weights = weights,
            solver = ACEfit.BLR())
    #test_rmse(results["errors"]["rmse"], rmse_blr, 1e-5)
end

@testset "BLR With Committee" begin
    rmse_blr = Dict(
         "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
         "set"           => Dict("V"=>0.0619346, "E"=>0.00238807, "F"=>0.121907),
         "dia"           => Dict("V"=>0.0333255, "E"=>0.00130242, "F"=>0.0255582),
         "liq"           => Dict("V"=>0.0345897, "E"=>0.000397724, "F"=>0.157461),
         "bt"            => Dict("V"=>0.0822944, "E"=>0.00322198, "F"=>0.062758),)
    acefit!(model, data;
            data_keys...,
            weights = weights,
            solver = ACEfit.BLR(factorization = :svd, committee_size = 10))
    #test_rmse(results["errors"]["rmse"], rmse_blr, 1e-5)
end
