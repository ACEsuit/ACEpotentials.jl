


using ACE1pack, Test, JuLIP, LazyArtifacts
using ACE1.Testing: println_slim
using JuLIP.Testing: print_tf

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

@info("Test multitransform (it's a bit trickier)")
multitransform_params = Dict(
    "cutoffs" => Dict(
       ("C", "C") => (1, 2),
       ("C", "H") => (1, 3),
       ("H", "H") => (1, 3)),
    "transforms" => Dict(
        ("C", "C") => Dict("type"=> "polynomial"),
        ("C", "H") => Dict("type"=> "polynomial"),
        ("H", "H") => Dict("type" => "polynomial")),
    "type" => "multitransform") 

out = fill_defaults!(multitransform_params, param_key="transform")
for (key, val) in out["transforms"]
    for extra_key in ["r0", "p"]
        print_tf(@test haskey(val, extra_key))
    end
end
println()

@info("Test filling in rpi_basis without \"type\" among rad_basis parameters")
rpi_params = Dict(
    "species" => "something",
    "N" => 2, 
    "maxdeg" => 2, 
    "rad_basis" => Dict(
        "rin" => 2.0,
        "rcut" => 7.0
    ))
out = fill_defaults!(rpi_params, param_key="rpi_basis")


@info("Test parsing string of tuple into tuple of strings in basis")
rpi_basis_params = Dict(
        "degree" => Dict(
            "type" => "sparseM",
            "Dn" => Dict("default"=> 1.0),
            "Dl" => Dict("default"=> 1.5),
            "Dd" => Dict(
                "default" => 10,
                "(4, C)" => 8)),
        "transform" => Dict(
            "cutoffs" => Dict("(C, C)" => "(1.0, 2.3)"),
            "transforms" => Dict(
                "(C, C)" => Dict(
                    "type" => "polynomial")),
            "type" => "multitransform"))

out = parse_basis_keys(rpi_basis_params)
print_tf(@test haskey(out["degree"]["Dd"], (4, "C")))
print_tf(@test haskey(out["transform"]["transforms"], ("C", "C")))
print_tf(@test haskey(out["transform"]["cutoffs"], ("C", "C")))
println()

@info("Test that all *params get filled in correctly on smallest allowed input.")

@info("fit_params")
minimal_params = Dict(
    "data" => Dict(
        "fname" => "something"),
    "basis" => Dict(
        "rpi_basis" => Dict(
            "species" => "something",
            "N" => 1,
            "maxdeg" => 1
            ),
        "pair_basis" => Dict(
            "species" => "sth",
            "maxdeg" => 1
            ),),
    "solver" => Dict(
        "solver" => "rrqr"),
    "e0" => "something")

filled_params = fill_defaults!(minimal_params)
for extra_key in ["ACE_fname", "LSQ_DB_fname_stem", "fit_from_LSQ_DB"]
    print_tf(@test haskey(filled_params, extra_key))
end
for extra_key in ["energy_key", "force_key", "virial_key"]
    print_tf(@test haskey(filled_params["data"], extra_key))
end
for extra_key in ["rad_basis", "transform", "degree"]
    print_tf(@test haskey(filled_params["basis"]["rpi_basis"], extra_key))
end
for extra_key in ["rcut", "rin", "pcut", "pin", "transform"]
    print_tf(@test haskey(filled_params["basis"]["pair_basis"], extra_key))
end
for extra_key in ["rrqr_tol"]
    print_tf(@test haskey(filled_params["solver"], extra_key))
end
println()

@info("The rest to be or not to be done...")


