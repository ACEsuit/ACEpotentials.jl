
using LazyArtifacts

data_dir = joinpath(artifact"TiAl_tiny_dataset", "data")
tests_files_dir = joinpath(artifact"ACE1pack_test_files", "files")

if !isfile(joinpath(data_dir, "TiAl_tiny.xyz"))
    zip = joinpath(artifact"TiAl_tiny_dataset", "TiAl_tiny.zip")
    run(`unzip $zip -d $data_dir`)
end

if !isfile(joinpath(tests_files_dir, "expected_fit_errors.json"))
    zip = joinpath(artifact"ACE1pack_test_files", "ACE1pack_test_files.zip")
    run(`unzip $zip -d $tests_files_dir`)
end