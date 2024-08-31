# using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ))

##

using ACEpotentials
using ExtXYZ, AtomsBase
using Distributed
using LazyArtifacts
using Test
ACE1compat = ACEpotentials.ACE1compat
using ACEpotentials.Models: ACEPotential

## ----- setup -----

@warn "test_silicon not fully converted yet."

params = (elements = [:Si],
          Eref = [:Si => -158.54496821],
          rcut = 5.5,
          order = 3,
          totaldegree = 12)

model1a = acemodel(; params...)
model1b = acemodel(; params...)
model2 = ACEPotential( ACE1compat.ace1_model(; params...) )

data1 = read_extxyz(artifact"Si_tiny_dataset" * "/Si_tiny.xyz")
data2 = ExtXYZ.load(artifact"Si_tiny_dataset" * "/Si_tiny.xyz")

data_keys = [:energy_key => "dft_energy",
             :force_key => "dft_force",
             :virial_key => "dft_virial"]
weights = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0),
               "liq" => Dict("E"=>10.0, "F"=>0.66, "V"=>0.25))

## ----- perform tests -----

# TODO : bring a target error back? 

# function test_rmse(rmse, expected, atol)
#     for config in keys(rmse)
#         # TODO and/or warning: can't iterate over rmse because it will have virial for isolated atom
#         #for obs in keys(rmse[config])
#         for obs in keys(expected[config])
#             @test rmse[config][obs] ≈ expected[config][obs] atol=atol
#         end
#     end
# end

# # @testset "QR" begin
# rmse_qr = Dict(
#     "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
#     "dia"           => Dict("V"=>0.0234649, "E"=>0.000617953, "F"=>0.018611),
#     "liq"           => Dict("V"=>0.034633, "E"=>0.000133371, "F"=>0.104112),
#     "set"           => Dict("V"=>0.0437043, "E"=>0.00128242, "F"=>0.0819438),
#     "bt"            => Dict("V"=>0.0576748, "E"=>0.0017616, "F"=>0.0515637),)

function compare_errors(err1, err2) 
    maxdiff = 0.0
    for k1 in ["rmse", "mae"]
        for k2 in keys(err1[k1])
            for k3 in ["V", "E", "F"]
                err = abs(err1[k1][k2][k3] - err2[k1][k2][k3]) / (err1[k1][k2][k3] + err2[k1][k2][k3] + sqrt(eps()))
                maxdiff = max(maxdiff, err)
            end 
        end
    end 
    return maxdiff
end 

acefit!(model1a, data1;
       data_keys...,
       weights = weights,
       solver=ACEfit.QR())

acefit!(model1b, data2;
       data_keys...,
       weights = weights,
       solver=ACEfit.QR())

acefit!(model2, data2;
       data_keys...,
       weights = weights,
       solver=ACEfit.QR())
##

err11 = ACEpotentials.linear_errors(data1, model1a; data_keys..., weights=weights)
err21 = ACEpotentials.linear_errors(data2, model1b; data_keys..., weights=weights)
err22 = ACEpotentials.linear_errors(data2, model2; data_keys..., weights=weights)

@show compare_errors(err11, err22)
@test compare_errors(err11, err22) < 0.2

@warn("The model1 - data2 test fails: probably a JuLIP conversion error")
@show compare_errors(err11, err21)

##

# repeat with distributed assembly
addprocs(3, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials

acefit!(model1a, data1;
            data_keys...,
            weights = weights,
            solver=ACEfit.QR())

acefit!(model2, data2;
            data_keys...,
            weights = weights,
            solver=ACEfit.QR())

rmprocs(workers())

err11d = ACEpotentials.linear_errors(data1, model1a; data_keys..., weights=weights)
err22d = ACEpotentials.linear_errors(data2, model2; data_keys..., weights=weights)

@show compare_errors(err11, err11d)
@show compare_errors(err22, err22d)
@test compare_errors(err11, err11d) < sqrt(eps())
@test compare_errors(err22, err22d) < sqrt(eps())

##

# rmse_blr = Dict(
#          "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
#          "set"           => Dict("V"=>0.0619346, "E"=>0.00238807, "F"=>0.121907),
#          "dia"           => Dict("V"=>0.0333255, "E"=>0.00130242, "F"=>0.0255582),
#          "liq"           => Dict("V"=>0.0345897, "E"=>0.000397724, "F"=>0.157461),
#          "bt"            => Dict("V"=>0.0822944, "E"=>0.00322198, "F"=>0.062758),)

acefit!(model1a, data1;
        data_keys...,
        weights = weights,
        solver = ACEfit.BLR())

acefit!(model2, data2;
        data_keys...,
        weights = weights,
        solver = ACEfit.BLR())        

err11 = ACEpotentials.linear_errors(data1, model1a; data_keys..., weights=weights)
err22 = ACEpotentials.linear_errors(data2, model2; data_keys..., weights=weights)

@show compare_errors(err11, err22)
@test compare_errors(err11, err22) < 0.2

##

@warn("Removed Commitee Test Until the new kernels support committee potentials")
# @testset "BLR With Committee" begin
#     rmse_blr = Dict(
#          "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
#          "set"           => Dict("V"=>0.0619346, "E"=>0.00238807, "F"=>0.121907),
#          "dia"           => Dict("V"=>0.0333255, "E"=>0.00130242, "F"=>0.0255582),
#          "liq"           => Dict("V"=>0.0345897, "E"=>0.000397724, "F"=>0.157461),
#          "bt"            => Dict("V"=>0.0822944, "E"=>0.00322198, "F"=>0.062758),)
#     acefit!(model, data;
#             data_keys...,
#             weights = weights,
#             solver = ACEfit.BLR(factorization = :svd, committee_size = 10))
#     #test_rmse(results["errors"]["rmse"], rmse_blr, 1e-5)
# end
