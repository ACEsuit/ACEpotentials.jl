


using ACEpotentials, Test, LazyArtifacts

test_train_set = joinpath(artifact"TiAl_tiny_dataset", "TiAl_tiny.xyz")

@info("Test constructing `data_params` and reading data")
params = data_params(fname = test_train_set, energy_key = "energy", force_key = "force", virial_key = "virial")
