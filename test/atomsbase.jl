using ACEpotentials
using Test

@testset "ACE model interface" begin
    data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial"; atoms_base=true)

    model = acemodel(
        elements = [:Ti, :Al],
        order = 3,
        totaldegree = 6,
        rcut = 5.5,
        Eref = [:Ti => -1586.0195, :Al => -105.5954]
    )


    weights = Dict(
        "FLD_TiAl" => Dict("E" => 60.0, "F" => 1.0 , "V" => 1.0 ),
        "TiAl_T5000" => Dict("E" => 5.0, "F" => 1.0 , "V" => 1.0 )
    );

    solver = ACEfit.LSQR(damp = 1e-2, atol = 1e-6);
    data_train = data[1:5:end]
    P = smoothness_prior(model; p = 4) 

    acefit!(model, data_train; solver=solver, weights=weights, prior = P, repulsion_restraint=true);
    ce, err = ACEpotentials.linear_errors(data, model; weights=weights);
    @test err["mae"]["F"] < 0.6
end