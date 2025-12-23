
module ETModels

# utility layers : these should likely be moved into ET or be removed 
#     if more convenient implementations can be found. 
#  
include("et_envbranch.jl")

# ET based ACE model components 
include("et_ace.jl")
include("onebody.jl")
include("et_pair.jl")

# converstion utilities: convert from 0.8 style ACE models to ET based models
include("convert.jl")

# utilities to convert radial embeddings to splined versions
# for simplicity and performance and to freeze parameters
include("splinify.jl")

include("et_calculators.jl")

end