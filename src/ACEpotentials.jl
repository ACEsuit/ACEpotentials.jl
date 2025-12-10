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
include("repulsion_restraint.jl")

# Data 
include("example_data.jl")

# Misc 
include("analysis/dataset_analysis.jl")
include("analysis/potential_analysis.jl")
include("descriptor.jl")


# TODO: to be completely rewritten or retired 
# include("export.jl")


# ----------------- Exports that seem important to make the tutorials work. 

import ACEpotentials.ACE1compat: ace1_model 
import ACEpotentials.Models: algebraic_smoothness_prior, 
                             exp_smoothness_prior, 
                             gaussian_smoothness_prior, 
                             set_parameters!, 
                             @committee, 
                             set_committee!
import JSON 

@deprecate linear_errors compute_errors 

export ace1_model,
       length_basis, 
       algebraic_smoothness_prior, 
       exp_smoothness_prior, 
       gaussian_smoothness_prior, 
       set_parameters!, 
       fast_evaluator, 
         @committee,
         set_committee!, 
         compute_errors, 
         linear_errors


include("json_interface.jl")


end
