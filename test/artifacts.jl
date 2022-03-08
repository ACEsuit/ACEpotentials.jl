
using LazyArtifacts

# data_dir = joinpath(artifact"TiAl_tiny_dataset", "data")
# tests_files_dir = joinpath(artifact"ACE1pack_test_files", "files")

# if !isdir(data_dir)
#     mkdir(data_dir)
# end

# if !isdir(tests_files_dir)
#     mkdir(tests_files_dir)
# end


# if !isfile(joinpath(artifact"ACE1pack_test_files", "TiAl_tiny.xyz"))
#     println("tarring things")
#     tar = joinpath(artifact"TiAl_tiny_dataset", "TiAl_tiny.tar.gz")
#     # run(`tar -xzvf $tar -C $data_dir`)
#     run(`tar -xzvf $tar`)

# end

# # if !isfile(joinpath(tests_files_dir, "expected_fit_errors.json"))
# if !isfile(joinpath(artifact"ACE1pack_test_files", "expected_fit_errors.json"))
#     tar = joinpath(artifact"ACE1pack_test_files", "ACE1pack_test_files.tar.gz")
#     # run(`tar -xzvf $tar -C $tests_files_dir`)
#     run(`tar -xzvf $tar `)
# end
