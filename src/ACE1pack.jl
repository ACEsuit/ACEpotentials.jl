module ACE1pack

# load and reexport JuLIP, ACE1, ACEfit 
using Reexport 
@reexport using JuLIP
@reexport using ACE1
@reexport using ACEfit
export JuLIP, ACE1, ACEfit 

# Convenience Layer 

include("acefit.jl")
include("acefit_helper.jl")

include("fit.jl")

include("data.jl")

include("basis.jl")

include("solver.jl")

include("regularizer.jl")

include("read_params.jl")

include("export.jl")

include("export_multispecies.jl")

# a little hack to load ACE1pack artifacts from anywhere? 
using LazyArtifacts
artifact(str) = (@artifact_str str)

end
