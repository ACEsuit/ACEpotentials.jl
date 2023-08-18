using ACEpotentials
using ArtifactUtils
using Pkg.Artifacts


function safe_add_artifact!(label, url, artifacts_toml)
    data_hash = artifact_hash(label, artifacts_toml)
    if data_hash == nothing || !artifact_exists(data_hash)
        add_artifact!(artifacts_toml, label, url, lazy=true, force=true)
    end
end


artifacts_toml = joinpath(pathof(ACEpotentials)[1:end-16], "Artifacts.toml")

label = "TiAl_tiny_dataset"
url = "https://github.com/ACEsuit/ACEData/blob/master/trainingsets/TiAl_tiny.tar.gz?raw=true"
safe_add_artifact!(label, url, artifacts_toml)


label = "ACEpotentials_test_files"
url = "https://github.com/ACEsuit/ACEData/blob/master/tests/ACEpotentials_test_files.tar.gz?raw=true"
safe_add_artifact!(label, url, artifacts_toml)

