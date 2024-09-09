
tmpproj = tempname()
ap_dir = joinpath(@__DIR__(), "..", "..")
using LazyArtifacts
datafile = artifact"Si_tiny_dataset" * "/Si_tiny.xyz"
julia_cmd = Base.julia_cmd()

run(`mkdir $tmpproj`)
run(`cp $datafile $(tmpproj*"/")`); 
prep_proj = "using Pkg; Pkg.activate(tmpproj); Pkg.develop(; path = ap_dir); using ACEpotentials; ACEpotentials.copy_runfit(tmpproj);"
run(`$( String(julia_cmd) * "-e '" * prep_proj * "'" )`)
# run(`$(Base.julia_cmd()) --project=$tmpproj $tmpproj/runfit.jl $tmpproj/example_params.json`)

##

# using ACEpotentials
# using ArtifactUtils
# using Pkg.Artifacts


# function safe_add_artifact!(label, url, artifacts_toml)
#     data_hash = artifact_hash(label, artifacts_toml)
#     if data_hash == nothing || !artifact_exists(data_hash)
#         add_artifact!(artifacts_toml, label, url, lazy=true, force=true)
#     end
# end


# artifacts_toml = joinpath(pathof(ACEpotentials)[1:end-16], "Artifacts.toml")

# label = "TiAl_tiny_dataset"
# url = "https://github.com/ACEsuit/ACEData/blob/master/trainingsets/TiAl_tiny.tar.gz?raw=true"
# safe_add_artifact!(label, url, artifacts_toml)


# label = "ACEpotentials_test_files"
# url = "https://github.com/ACEsuit/ACEData/blob/master/tests/ACEpotentials_test_files.tar.gz?raw=true"
# safe_add_artifact!(label, url, artifacts_toml)
