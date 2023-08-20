
using ACE1.Testing: println_slim

tutorials = [
   "TiAl_model.jl",
   "first_example_model.jl",
   "smoothness_priors.jl",
   "TiAl_basis.jl", 
   "descriptor.jl", 
   "dataset_analysis.jl",]

for tut in tutorials
   try
      include(joinpath(@__DIR__(), "..", "tutorials", tut))
      @info("No error thrown in tutorial $tut")
      println_slim(@test true)
   catch 
      @error("Error thrown in tutorial $tut")
      println_slim(@test false)
   end
end  