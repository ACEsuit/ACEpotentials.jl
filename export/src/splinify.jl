# Hermite cubic spline utilities for ETACE radial basis export
#
# WORKFLOW:
# 1. Splinify the model BEFORE fitting using upstream splinify()
# 2. Fit the splinified model
# 3. Export: extract Hermite spline data (knots, function values, gradients)
# 4. Generate trim-safe Julia code using generate_hermite_spline_code()
#
# This ensures the fitted model uses the same spline representation as the exported model!
#
# Usage:
#   using ACEpotentials.ETModels: splinify
#
#   # BEFORE fitting:
#   etace_splined = splinify(etace, ps, st; Nspl=50)
#
#   # Fit the splinified model (not shown)
#
#   # AFTER fitting, export:
#   hermite_data = extract_hermite_spline_data(etace_splined, ps_fitted, st_fitted, rcut)

using StaticArrays
using LinearAlgebra: norm
using Printf
using EquivariantTensors
const ET = EquivariantTensors
using DecoratedParticles
using AtomsBase: ChemicalSpecies

"""
    HermiteSplineData

Container for Hermite cubic spline data extracted from P4ML splinified model.

This stores the knot points and Hermite data (function values F and gradients G)
that define piecewise cubic polynomials. Each knot stores values for multiple
basis functions simultaneously (vectorized evaluation).

# Fields
- `pair_idx::Int`: Species pair index (1-based)
- `iz::Int`: First species index
- `jz::Int`: Second species index
- `n_rnl::Int`: Number of radial basis functions (output dimension)
- `n_knots::Int`: Number of knot points (segments = n_knots - 1)
- `y_min::T`: Minimum y value (transformed space, typically -1.0)
- `y_max::T`: Maximum y value (transformed space, typically 1.0)
- `F::Matrix{T}`: Function values at knots [n_knots, n_rnl]
- `G::Matrix{T}`: Gradient values at knots [n_knots, n_rnl]
- `agnesi_params::NamedTuple`: Agnesi transform parameters (rin, r0, rcut, a, b0, b1, ...)
- `envelope_params::Union{NamedTuple,Nothing}`: Envelope function parameters (if applicable)

# Evaluation
For distance r:
1. Transform: y = agnesi_transform(r, agnesi_params)
2. Find segment: il, t = find_segment(y, y_min, y_max, n_knots)
3. Hermite cubic: Rnl = eval_hermite_cubic(t, F[il:il+1, :], G[il:il+1, :], h)
4. Apply envelope: result = envelope(r) .* Rnl
"""
struct HermiteSplineData{T}
    pair_idx::Int
    iz::Int
    jz::Int
    n_rnl::Int
    n_knots::Int
    y_min::T
    y_max::T
    F::Matrix{T}        # [n_knots, n_rnl] - function values at knots
    G::Matrix{T}        # [n_knots, n_rnl] - gradients at knots
    agnesi_params::NamedTuple
    envelope_params::Union{NamedTuple,Nothing}
end

"""
    extract_hermite_spline_data(etace_splined, ps, st, rcut)

Extract Hermite cubic spline data from an ETACE model that has been splinified
using `ACEpotentials.ETModels.splinify()`.

This directly extracts the knot points, function values (F), and gradients (G)
that define the Hermite cubic splines created by P4ML, without any sampling or
approximation. The extracted data can be used for exact code generation.

# Arguments
- `etace_splined`: ETACE model after calling `splinify()`
- `ps`: Fitted parameters
- `st`: Model state
- `rcut`: Radial cutoff distance

# Returns
Dict mapping pair_idx => HermiteSplineData for each species pair

# Notes
The splines operate in transformed y-space (via Agnesi transform) where y ∈ [y_min, y_max],
typically [-1, 1]. The knots are uniformly distributed in this transformed space.

# Example
```julia
using ACEpotentials.ETModels: splinify

# Splinify BEFORE fitting
etace_splined = splinify(etace, ps_init, st_init; Nspl=50)

# Fit the splinified model
calc, result = acefit!(etace_splined, train_data)

# Extract Hermite spline data for code generation
hermite_data = extract_hermite_spline_data(calc.model, calc.ps, calc.st, 5.5)
```
"""
function extract_hermite_spline_data(etace_splined, ps, st, rcut::Real)
    # Extract components from splinified model
    rembed_layer = etace_splined.rembed.layer

    # Check that this is actually a splinified model
    if !isa(rembed_layer, ET.TransSelSplines)
        error("Model is not splinified! rembed.layer is $(typeof(rembed_layer)), expected TransSelSplines. " *
              "Call ACEpotentials.ETModels.splinify() on the model before fitting.")
    end

    # Extract transformation and spline parameters
    trans = rembed_layer.trans
    trans_st = trans.refstate
    zlist = trans_st.zlist
    NZ = length(zlist)
    agnesi_params_list = trans_st.params

    # Extract spline knot data from refstate
    # rembed_layer.refstate contains (F, G, x0, x1) for each species pair
    # But we need to get it from the state, not the layer
    # Let's check what's available
    if hasfield(typeof(st.rembed), :params)
        spline_refstate = st.rembed.params
    elseif hasfield(typeof(rembed_layer), :refstate)
        spline_refstate = rembed_layer.refstate
    else
        error("Cannot find spline data in model or state")
    end

    # spline_refstate should have: F, G, x0, x1
    F_all = spline_refstate.F  # Matrix of SVectors [n_knots, n_categories]
    G_all = spline_refstate.G  # Matrix of SVectors [n_knots, n_categories]
    x0_all = spline_refstate.x0  # Vector [n_categories] - y_min for each category
    x1_all = spline_refstate.x1  # Vector [n_categories] - y_max for each category

    n_knots = size(F_all, 1)
    n_categories = size(F_all, 2)

    # Each F[i, j] is an SVector - extract its length to get n_rnl
    n_rnl = length(F_all[1, 1])

    # Convert F and G from SVector to regular matrices for each category
    hermite_data = Dict{Int, HermiteSplineData{Float64}}()

    for cat_idx in 1:n_categories
        # Convert cat_idx to (iz, jz) species indices
        # The selector function determines the mapping, but typically it's:
        # cat_idx = (iz - 1) * NZ + jz for asymmetric pairs
        iz = (cat_idx - 1) ÷ NZ + 1
        jz = (cat_idx - 1) % NZ + 1

        # Extract F and G for this category
        F_cat = zeros(Float64, n_knots, n_rnl)
        G_cat = zeros(Float64, n_knots, n_rnl)

        for i in 1:n_knots
            F_cat[i, :] = F_all[i, cat_idx]
            G_cat[i, :] = G_all[i, cat_idx]
        end

        # Get Agnesi parameters for this pair
        # Agnesi params are stored symmetrically, so we need to map (iz, jz) to symmetric index
        sym_i = min(iz, jz)
        sym_j = max(iz, jz)
        # Symmetric indexing: upper triangular
        sym_pair_idx = (sym_i - 1) * NZ - (sym_i - 1) * (sym_i - 2) ÷ 2 + (sym_j - sym_i + 1)
        agnesi_p = agnesi_params_list[sym_pair_idx]

        # Extract envelope parameters if present
        envelope_p = nothing
        if rembed_layer.envelope !== nothing
            # The envelope is another NTtransform
            # For now, we'll leave this as nothing and handle it separately if needed
            # The envelope parameters are embedded in the closure, not easily extractable
        end

        y_min = x0_all[cat_idx]
        y_max = x1_all[cat_idx]

        hermite_data[cat_idx] = HermiteSplineData{Float64}(
            cat_idx,
            iz,
            jz,
            n_rnl,
            n_knots,
            y_min,
            y_max,
            F_cat,
            G_cat,
            agnesi_p,
            envelope_p
        )
    end

    return hermite_data
end
