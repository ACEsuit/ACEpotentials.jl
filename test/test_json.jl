
tmpproj = tempname()
ap_dir = joinpath(@__DIR__(), "..")
using LazyArtifacts
datafile = artifact"Si_tiny_dataset" * "/Si_tiny.xyz"
julia_cmd = Base.julia_cmd()

# prepare the project folder 
run(`mkdir $tmpproj`)
run(`cp $datafile $(tmpproj*"/")`);
prep_proj = """
   begin 
      using Pkg; 
      Pkg.activate(\"$tmpproj\"); 
      Pkg.develop(; path = \"$ap_dir\"); 
      using ACEpotentials; 
      ACEpotentials.copy_runfit(\"$tmpproj\"); 
   end
"""
run(`$julia_cmd -e $prep_proj`)

# run the fit 
cd(tmpproj) do 
   run(`pwd`)
   run(`$julia_cmd --project=. runfit.jl -p example_params.json`)
end

# load the results in the current process! 
@info("Load the results")
using JSON 
example_params = JSON.parsefile(joinpath(tmpproj, "example_params.json"))
results = JSON.parsefile(joinpath(tmpproj, "example_params_results/results.json"))

@info("Clean up temporary project")
run(`rm -rf $tmpproj`)

##

using ACEpotentials

@info("Check that runfit gives the same results as a julia script")
@info("TODO: ")

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
