
include("artifacts.jl")

@testset "Read data" begin

    using ACE1pack

    test_train_set = joinpath(artifact"TiAl_tiny_dataset", "TiAl_tiny.xyz")

    @info("Test constructing `data_params` and reading data")
    params = data_params(fname = test_train_set, energy_key = "energy", force_key = "force", virial_key = "virial")
    data = ACE1pack.read_data(params)

end 
