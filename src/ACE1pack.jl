module ACE1pack

using Reexport 
@reexport using ACE1
@reexport using ACE1x
@reexport using ACEfit
@reexport using JuLIP
export ACE1, ACE1x, ACEfit, JuLIP

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

# hack to load ACE1pack artifacts from anywhere? 
using LazyArtifacts
artifact(str) = (@artifact_str str)

end
