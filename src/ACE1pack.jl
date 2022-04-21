module ACE1pack

# load and reexport JuLIP, ACE1, ACEfit 
using Reexport 
@reexport using JuLIP
@reexport using ACE1
@reexport using ACEfit
export JuLIP, ACE1, ACEfit 

# Convenience Layer 

include("acefit.jl")

include("fit.jl")

include("data.jl")

include("basis.jl")

include("solver.jl")

include("precon.jl")

include("read_params.jl")


end
