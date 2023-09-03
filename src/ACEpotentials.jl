module ACEpotentials

using Reexport 
@reexport using JuLIP
@reexport using ACE1
@reexport using ACE1x
@reexport using ACEfit
@reexport using ACEmd

include("atoms_data.jl")
include("fit.jl")
include("data.jl")
include("basis.jl")
include("model.jl")
include("solver.jl")
include("regularizer.jl")
include("read_params.jl")
include("export.jl")
include("descriptor.jl")

include("analysis/potential_analysis.jl")
include("analysis/dataset_analysis.jl")

include("example_data.jl")

end
