# using Pkg; Pkg.activate(joinpath(@__DIR__(), "..", ))

##

using ACEpotentials
using ACEbase.Testing: println_slim
using ExtXYZ, AtomsBase
using Distributed
using LazyArtifacts
using Test
ACE1compat = ACEpotentials.ACE1compat
using ACEpotentials.Models: ACEPotential

## ----- setup -----

params = (elements = [:Si],
          Eref = [:Si => -158.54496821],
          rcut = 5.5,
          order = 3,
          totaldegree = 12)

model = ACE1compat.ace1_model(; params...) 

data = ExtXYZ.load(artifact"Si_tiny_dataset" * "/Si_tiny.xyz")

data_keys = [:energy_key => "dft_energy",
             :force_key  => "dft_force",
             :virial_key => "dft_virial"]

weights = Dict("default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0),
                   "liq" => Dict("E"=>10.0, "F"=>0.66, "V"=>0.25))

## ----- perform tests -----

function test_rmse(rmse, expected)
    for config in keys(rmse)
        # TODO and/or warning: can't iterate over rmse because it will have virial for isolated atom
        #for obs in keys(rmse[config])
        for obs in keys(expected[config])
            @test rmse[config][obs] <= expected[config][obs]
        end
    end
end

rmse_qr = Dict(
    "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
    "dia"           => Dict("V"=>0.027, "E"=>0.0012, "F"=>0.024),
    "liq"           => Dict("V"=>0.037, "E"=>0.0006, "F"=>0.16),
    "set"           => Dict("V"=>0.057, "E"=>0.0017, "F"=>0.12),
    "bt"            => Dict("V"=>0.08, "E"=>0.0022, "F"=>0.07),)

acefit!(data, model; 
       data_keys...,
       weights = weights,
       solver=ACEfit.QR())

err = ACEpotentials.linear_errors(data, model; data_keys..., weights=weights)

test_rmse(err["rmse"], rmse_qr)

##

# repeat with distributed assembly
addprocs(3, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials

acefit!(data, model;
            data_keys...,
            weights = weights,
            solver=ACEfit.QR())

rmprocs(workers())

err_dist = ACEpotentials.linear_errors(data, model; data_keys..., weights=weights)
test_rmse(err_dist["rmse"], rmse_qr)

##

rmse_blr = Dict(
         "isolated_atom" => Dict("E"=>0.0, "F"=>0.0),
         "set" => Dict("V"=>0.068, "E"=>0.0028, "F"=>0.14),
         "dia" => Dict("V"=>0.0333255, "E"=>0.0016, "F"=>0.03),
         "liq" => Dict("V"=>0.035, "E"=>0.00035, "F"=>0.19),
         "bt"  => Dict("V"=>0.09, "E"=>0.0036, "F"=>0.073),)


acefit!(data, model;
        data_keys...,
        weights = weights,
        solver = ACEfit.BLR())        

err_blr = ACEpotentials.linear_errors(data, model; data_keys..., weights=weights)

test_rmse(err_blr["rmse"], rmse_blr)


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


##
# Add a descriptor test 

using AtomsBuilder
sys = rattle!(bulk(:Si, cubic=true) * 2, 0.1)
X = site_descriptors(sys, model)
X234 = site_descriptors(sys, model; domain = [2,3,4])
println_slim( @test X234 == X[2:4] )


##
# rerun fit with repulsion restraint 

@info("Run a fit with repulsion restraint")

acefit!(data, model;
        data_keys...,
        weights = weights,
        solver = ACEfit.BLR(), 
        repulsion_restraint = true)

r = 0.001 + 0.01 * rand()
zSi = atomic_number(ChemicalSpecies(:Si))
Bpair = model.model.pairbasis(r, zSi, zSi, NamedTuple(), NamedTuple())
V2 = sum(model.ps.Wpair[:, 1] .* Bpair)
println_slim(@test V2 > 10_000)
