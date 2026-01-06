# Trim-safe evaluation kernels for ACE model export
#
# This module provides generic, parameterized functions for:
# - Polynomial basis evaluation (3-term recurrence)
# - Distance transforms (Agnesi, normalized)
# - Envelope functions (quartic, polynomial)
# - Hermite cubic spline interpolation
#
# These kernels are designed to be included in exported models, replacing
# per-pair code generation with parameterized table lookups.
#
# Usage:
#   include("kernels/kernels.jl")
#   # Then in exported model code:
#   eval_polys_generic!(P, POLY_A, POLY_B, POLY_C, y)
#   agnesi_transform(r, TRANSFORM_PARAMS[pair_idx])
#   envelope_quartic(y)
#   hermite_eval(y, y_min, y_max, h, F, G)

# Include all kernel files
include("polynomials.jl")
include("transforms.jl")
include("envelopes.jl")
include("hermite.jl")
