
@testset "Read params" begin

using ACE1pack, JuLIP

    include("artifacts.jl")
    test_train_set = joinpath(data_dir, "TiAl_tiny.xyz")
    json_params_fname = joinpath(tests_files_dir, "fit_params.json")
    yaml_params_fname = joinpath(tests_files_dir, "fit_params.yaml")

    data = Dict(
        "energy_key"   => "energy",
        "fname" => test_train_set,
        "virial_key"   => "virial")

    @info("Quick test for filling in missing param entries with defaults")
    data = ACE1pack.fill_defaults!(data, param_key = "data")
    @test "force_key" in collect(keys(data))

    @info("Test loading params from json")
    fit_params = load_dict(json_params_fname)
    fit_params["data"]["fname"] = test_train_set 
    fit_params = fill_defaults!(fit_params)

    @info("Test loading params from yaml")
    fit_params = load_dict(yaml_params_fname)
    fit_params["data"]["fname"] = test_train_set 
    fit_params = fill_defaults!(fit_params)


end

