
using LazyArtifacts

# Meta-data for these datasets are stored in the ACEdata  repository 
#   https://github.com/ACEsuit/ACEData
#
#


_example_data = Dict(
   "Si_tiny" => (  art = "Si_tiny_dataset", 
                 train = "Si_tiny.xyz", 
                  test = nothing, 
                  meta = "" ), 
   "TiAl_tiny" => ( art = "TiAl_tiny_dataset", 
                  train = "TiAl_tiny.xyz", 
                   test = nothing, 
                   meta = "" ), 
   "TiAl_tutorial" => ( art = "TiAl_tutorial", 
                      train = "TiAl_tutorial.xyz", 
                       test = nothing, 
                       meta = "" ),
     [ 
   "Zuo20_$(sym)" => ( art = "ZuoEtAl2020", 
                     train = "ZuoEtAl2020/$(sym)_train.xyz", 
                      test = "ZuoEtAl2020/$(sym)_test.xyz", 
                      meta = "data converted from Zuo et al, J Phys Chem A 124 (2020)" )
            for sym in [:Ni, :Cu, :Li, :Mo, :Si, :Ge]  ]... 
   )

function list_example_datasets()
   return sort(collect(keys(_example_data)))
end   

function example_dataset(id::Union{String, Symbol})
   art = _example_data[id].art
   path = (@artifact_str "$art")
   train_path = joinpath(path, _example_data[id].train)
   train = read_extxyz(train_path)
   if _example_data[id].test != nothing
      test_path = joinpath(path, _example_data[id].test)
      test = read_extxyz(test_path)
   else
      test = nothing
   end
   return (train = train, 
            test = test, 
            meta = _example_data[id].meta)
end 
   

