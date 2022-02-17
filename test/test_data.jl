
@testset "Read data" begin

    using ACE1pack

    include("artifacts.jl")
    test_train_set = joinpath(data_dir, "TiAl_tiny.xyz")

    @info("Test constructing `data_params` and reading data")
    params = data_params(fname = test_train_set, energy_key = "energy", force_key = "force", virial_key = "virial")
    data = ACE1pack.read_data(params)

end 
