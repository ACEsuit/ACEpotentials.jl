module ACEpotentials

using Unitful, AtomsBase, AtomsCalculators, Reexport

@reexport using ACEfit 

# TODO: make a list of exports while re-writing the docs and tutorials 
#       should we re-export ACEfit? I'm not convinced. 


include("defaults.jl")

include("models/models.jl")

include("ace1_compat.jl")

# include("model.jl")
# include("example_data.jl")
# include("descriptor.jl")

# TODO: to be completely rewritte
# include("io.jl")

# TODO: all of this just needs to be moved from JuLIP to AtomsBase
# include("analysis/potential_analysis.jl")
# include("analysis/dataset_analysis.jl")

# TODO: this is basically the UFACE interface which we need to revive
# include("experimental.jl")

end
