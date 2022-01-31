module ACE1pack

# load and reexport JuLIP, ACE1, IPFitting 
using Reexport 
@reexport using JuLIP
@reexport using ACE1
@reexport using IPFitting
export JuLIP, ACE1, IPFitting 

# Convenience Layer 

include("fit.jl")

include("data.jl")

include("basis.jl")

include("solver.jl")

include("read_params.jl")

end
