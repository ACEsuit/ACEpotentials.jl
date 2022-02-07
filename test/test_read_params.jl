
@testset "Read params" begin

using ACE1pack, JuLIP

    include("artifacts.jl")
    test_train_set = joinpath(data_dir, "TiAl_tiny.xyz")
    json_params = joinpath(tests_files_dir, "fit_params.json")

    data = Dict(
        "energy_key"   => "energy",
        "fname" => test_train_set,
        "virial_key"   => "virial")

    @info("Quick test for filling in missing param entries with defaults")
    data = ACE1pack.fill_defaults!(data, param_key = "data")
    @test "force_key" in collect(keys(data))

    @info("Test loading params from json")
    fit_params = fill_defaults!(load_dict(json_params))

    # TODO introduce once JuLIP has a read_yaml
    # @info("Test loading params from yaml")
    # fit_params = ACE1pack.yaml_to_params(@__DIR__() * "/files/fit_params.yaml")

end

