
module Models 

using Random: AbstractRNG

using StrideArrays, Bumper, WithAlloc
import WithAlloc: whatalloc

import Zygote

import Polynomials4ML
const P4ML = Polynomials4ML

import EquivariantTensors

import LuxCore: AbstractLuxLayer,
               AbstractLuxContainerLayer,
               initialparameters,
               initialstates      
               
function length_basis end
function energy_forces_virial_basis end
function potential_energy_basis end

include("elements.jl")

include("onebody.jl")
include("stacked_pot.jl")

include("radial_envelopes.jl")

include("radial_transforms.jl")

include("Rnl_basis.jl")
include("Rnl_learnable.jl")
include("Rnl_splines.jl")

# sparse.jl removed - now using EquivariantTensors.SparseACEbasis directly

include("ace_heuristics.jl") 
include("ace.jl")

include("calculators.jl")
include("committee.jl")

include("smoothness_priors.jl")

include("utils.jl")

# include("fasteval.jl")




end 