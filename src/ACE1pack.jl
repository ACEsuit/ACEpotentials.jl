module ACE1pack

# load and reexport JuLIP, ACE1
# also ArgParse and PyCall for the command line script
using Reexport 
@reexport using JuLIP
@reexport using ASE
@reexport using ACE1
@reexport using ACE1x
@reexport using ACEfit
@reexport using ArgParse
@reexport using PyCall
export JuLIP, ASE, ACE1, ACEfit, Argparse, PyCall

# Convenience Layer 

include("atoms_data.jl")

include("fit.jl")

include("data.jl")

include("basis.jl")

include("model.jl")

include("solver.jl")

include("regularizer.jl")

include("read_params.jl")

include("export.jl")

include("analysis.jl")

# a little hack to load ACE1pack artifacts from anywhere? 
using LazyArtifacts
artifact(str) = (@artifact_str str)

end
