
@testset "Read params" begin

using ACE1pack

data = Dict(
    "energy_key"   => "energy",
    "xyz_filename" => "/Users/elena/.julia/dev/ACE1pack/test/files/TiAl_tutorial_DB_tenth.xyz",
    "virial_key"   => "virial")

@info("Quick test for filling in missing param entries with defaults")
data = ACE1pack.fill_default_params(data, "data_params")
@test "force_key" in collect(keys(data))

@info("Test loading params from json")
fit_params = ACE1pack.json_to_params(@__DIR__() * "/files/fit_params.json")

@info("Test loading params from yaml")
fit_params = ACE1pack.yaml_to_params(@__DIR__() * "/files/fit_params.yaml")



end



