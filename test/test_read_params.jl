


using ACE1pack, Test, JuLIP, LazyArtifacts
using ACE1.Testing: println_slim

test_train_set = joinpath(artifact"TiAl_tiny_dataset", "TiAl_tiny.xyz")
json_params_fname = joinpath(artifact"ACE1pack_test_files", "fit_params.json")
yaml_params_fname = joinpath(artifact"ACE1pack_test_files", "fit_params.yaml")

data = Dict(
    "energy_key"   => "energy",
    "fname" => test_train_set,
    "virial_key"   => "virial")

@info("Quick test for filling in missing param entries with defaults")
data = ACE1pack.fill_defaults!(data, param_key = "data")
println_slim(@test "force_key" in collect(keys(data)))

@info("Test loading params from json")
fit_parms = load_dict(json_params_fname)
fit_parms["data"]["fname"] = test_train_set 
fit_parms = fill_defaults!(fit_parms)

@info("Test loading params from yaml")
fit_parms = load_dict(yaml_params_fname)
fit_parms["data"]["fname"] = test_train_set 
fit_parms = fill_defaults!(fit_parms)

