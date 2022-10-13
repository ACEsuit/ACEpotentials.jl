module ACE1pack

# load and reexport JuLIP, ACE1, IPFitting 
# also ArgParse and PyCall for the command line script
using Reexport 
@reexport using JuLIP
@reexport using ACE1
@reexport using IPFitting
@reexport using ArgParse
@reexport using PyCall
export JuLIP, ACE1, IPFitting, Argparse, PyCall

# Convenience Layer 

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
