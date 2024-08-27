module ACEpotentials

using Reexport 
@reexport using JuLIP

using ACE1
export ACE1 

using ACE1x
export ACE1x 
import ACE1x: ace_basis, smoothness_prior, ace_defaults, acemodel 
export ace_basis, smoothness_prior, ace_defaults, acemodel

using Unitful

@reexport using ACEfit
# @reexport using ACEmd

include("defaults.jl")

include("models/models.jl")
include("ace1_compat.jl")

include("atoms_data.jl")
include("model.jl")
include("export.jl")
include("example_data.jl")
include("descriptor.jl")
# include("atoms_base.jl")
# include("io.jl")

include("analysis/potential_analysis.jl")
include("analysis/dataset_analysis.jl")

include("experimental.jl")

# include("outdated/fit.jl")
# include("outdated/data.jl")
# include("outdated/basis.jl")
# include("outdated/solver.jl")
# include("outdated/regularizer.jl")
# include("outdated/read_params.jl")

end
