using Test
using ACEpotentials
using LazyArtifacts


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


@testset "IO for new save and load" begin
    acefit!(model, data;
            data_keys...,
            weights = weights,
            solver = ACEfit.LSQR(
                damp = 2e-2,
                conlim = 1e12, 
                atol = 1e-7,
                maxiter = 100000, 
                verbose = false
            )
    )
    fname = tempname() * ".json" 
    pot = ACEpotential(model.potential.components)
    @test_throws AssertionError save_potential(fname, model; meta="meta test")
    save_potential(fname, model; meta=Dict("test"=>"meta test") )
    npot = load_potential(fname; new_format=true)
    @test ace_energy(pot, data[1]) ≈ ace_energy(npot, data[1])
end