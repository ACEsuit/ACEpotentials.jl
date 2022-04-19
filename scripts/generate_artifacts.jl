using ACE1pack
using ArtifactUtils
using Pkg.Artifacts


function safe_add_artifact!(label, url, artifacts_toml)
    data_hash = artifact_hash(label, artifacts_toml)
    if data_hash == nothing || !artifact_exists(data_hash)
        add_artifact!(artifacts_toml, label, url, lazy=true, force=true)
    end
end


artifacts_toml = joinpath(pathof(ACE1pack)[1:end-16], "Artifacts.toml")

label = "TiAl_tiny_dataset"
url = "https://github.com/ACEsuit/ACEData/blob/master/trainingsets/TiAl_tiny.tar.gz?raw=true"
safe_add_artifact!(label, url, artifacts_toml)


label = "ACE1pack_test_files"
url = "https://github.com/ACEsuit/ACEData/blob/master/tests/ACE1pack_test_files.tar.gz?raw=true"
safe_add_artifact!(label, url, artifacts_toml)

