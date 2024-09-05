module ACEpotentials

using Unitful, AtomsBase, AtomsCalculators, Reexport

@reexport using ACEfit 

# TODO: make a list of exports while re-writing the docs and tutorials 
#       should we re-export ACEfit? I'm not convinced. 

# Constructing models 
include("defaults.jl")
include("models/models.jl")
include("ace1_compat.jl")

# Fitting
include("atoms_data.jl")
include("fit_model.jl")

# Data 
include("example_data.jl")

# Misc 
# TODO: all of this just needs to be moved from JuLIP to AtomsBase
# include("descriptor.jl")
# include("analysis/potential_analysis.jl")
# include("analysis/dataset_analysis.jl")

# TODO: to be completely rewritten
# include("io.jl")
# include("export.jl")

# Experimental 
# TODO: this is basically the UFACE interface which we need to revive
# include("experimental.jl")

end
