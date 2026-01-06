# Export an ETACE model to trim-compatible Julia code
#
# Usage:
#   include("export_ace_model.jl")
#   export_ace_model(calc, "my_model.jl")
#
# The generated file can be compiled with:
#   juliac --output-lib libace.so --trim=safe model.jl

using ACEpotentials
using ACEpotentials.ETModels: ETACEPotential, ETACE, StackedCalculator, ETOneBody
using StaticArrays
using SparseArrays
using LinearAlgebra
using Polynomials4ML
const P4ML = Polynomials4ML
using EquivariantTensors
const ET = EquivariantTensors
using AtomsBase: ChemicalSpecies

# Include spline utilities and code generators
include("splinify.jl")
include("codegen.jl")


"""
    export_ace_model(calc::StackedCalculator, filename::String; kwargs...)

Export a StackedCalculator (e.g., ETOneBody + ETACE) to a trim-compatible Julia file.

Automatically extracts E0 values from any ETOneBody calculators and the main ETACE model.
"""
function export_ace_model(calc::StackedCalculator, filename::String; kwargs...)
    # Find ETOneBody and ETACE calculators in the stack
    e0_calc = nothing
    etace_calc = nothing

    for subcalc in calc.calcs
        if isa(subcalc.model, ETOneBody)
            e0_calc = subcalc
        elseif isa(subcalc.model, ETACE)
            etace_calc = subcalc
        end
    end

    if etace_calc === nothing
        error("StackedCalculator must contain an ETACE model")
    end

    # Extract E0 values if ETOneBody is present
    E0_dict = nothing
    if e0_calc !== nothing
        E0s = e0_calc.st.E0s  # SVector of E0 values
        categories = e0_calc.model.categories  # Species
        E0_dict = Dict(Int(cat.atomic_number) => E0s[i] for (i, cat) in enumerate(categories))
        @info "Extracted E0 values from ETOneBody" E0_dict
    end

    # Call the main export function with E0 values
    return export_ace_model(etace_calc, filename; E0_dict=E0_dict, kwargs...)
end


"""
    export_ace_model(calc::ETACEPotential, filename::String; for_library=false, radial_basis=:polynomial)

Export an ETACEPotential to a trim-compatible Julia file.

Arguments:
- `calc`: The fitted ETACEPotential to export
- `filename`: Output filename
- `for_library=false`: If true, generate a shared library with C interface instead of executable
- `radial_basis=:polynomial`: Radial basis evaluation method
  - `:polynomial` - Runtime polynomial evaluation (default, exact, works with any model)
  - `:hermite_spline` - Hermite cubic splines (fast, exact). Model must be pre-splinified.

# Example 1: Standard polynomial export (no pre-processing needed)
```julia
calc = ETACEPotential(etace, ps_fitted, st_fitted, 5.5)
export_ace_model(calc, "my_model.jl"; for_library=true)
# Compile with: juliac --output-lib libace.so --trim=safe my_model.jl
```

# Example 2: Fast Hermite spline export (requires pre-splinification)
```julia
using ACEpotentials.ETModels: splinify

# Splinify BEFORE fitting for ~3-4x faster evaluation
etace_splined = splinify(etace, ps, st; Nspl=50)
# ... fit the splinified model ...
calc = ETACEPotential(etace_splined, ps_fitted, st_fitted, 5.5)

export_ace_model(calc, "my_model.jl"; for_library=true, radial_basis=:hermite_spline)
```

# Performance
- `:polynomial` - Evaluates Chebyshev polynomials at runtime (~585 ops/neighbor)
- `:hermite_spline` - Table lookup + cubic interpolation (~150 ops/neighbor, ~3-4x faster)

# Notes
- `:hermite_spline` provides exact reproduction of P4ML splines with C1 continuity
- `:polynomial` works with any model but is slower due to runtime polynomial evaluation
"""
function export_ace_model(calc::ETACEPotential, filename::String;
                          for_library::Bool=false,
                          radial_basis::Symbol=:polynomial,
                          E0_dict::Union{Dict{Int,Float64},Nothing}=nothing)

    # Extract ETACE components from the calculator
    # WrappedSiteCalculator has fields: model, ps, st, rcut
    etace = calc.model
    ps = calc.ps
    st = calc.st
    rcut = calc.rcut

    # EdgeEmbed wraps EmbedDP which has: trans, basis, post
    # etace.rembed is EdgeEmbed, etace.rembed.layer is EmbedDP
    rembed_layer = etace.rembed.layer  # EmbedDP{NTtransformST, WrappedBasis, SelectLinL}
    yembed_layer = etace.yembed.layer  # EmbedDP{NTtransformST, RealSCWrapper, IDpost}

    # Auto-detect if model is splinified
    is_splinified = isa(rembed_layer, ET.TransSelSplines)

    # Validate radial_basis choice against model state
    if radial_basis == :hermite_spline && !is_splinified
        @warn "Model is not splinified, but radial_basis=:hermite_spline requires splinification. Falling back to :polynomial evaluation."
        radial_basis = :polynomial
    elseif radial_basis == :polynomial && is_splinified
        @warn "Model is splinified, but radial_basis=:polynomial requires original polynomial structure. " *
              "Splinification replaces polynomials with splines. Using :hermite_spline instead."
        radial_basis = :hermite_spline
    end

    # Extract species information from radial embedding state
    # NTtransformST has fields: f, refstate (not st)
    trans_st = rembed_layer.trans.refstate
    zlist = trans_st.zlist  # Tuple of ChemicalSpecies
    _i2z = [Int(z.atomic_number) for z in zlist]
    NZ = length(_i2z)

    # Extract Agnesi transform parameters
    agnesi_params = trans_st.params  # SVector of Agnesi parameter NamedTuples

    # Tensor components (same structure as old ACE)
    tensor = etace.basis

    # Spherical harmonics - extract maxl from the yembed layer
    # yembed_layer.basis is RealSCWrapper{SolidHarmonics}
    ybasis = yembed_layer.basis
    maxl = P4ML.maxl(ybasis)

    # Radial basis spec and weights
    # Different structure for splinified vs non-splinified models
    if is_splinified
        # TransSelSplines structure: trans, envelope, selector, refstate
        # Extract n_rnl from spline data (F matrix has SVector{n_rnl, ...} elements)
        spline_refstate = st.rembed.params
        F_sample = spline_refstate.F[1, 1]  # Get first SVector to determine size
        n_rnl = length(F_sample)
        n_polys = nothing  # Not applicable for splines
        poly_basis = nothing
        W_radial = nothing  # Splines don't use weight matrix
    else
        # EmbedDP structure: trans -> basis (WrappedBasis) -> post (SelectLinL)
        rbasis_linl = rembed_layer.post  # The SelectLinL layer
        n_polys = rbasis_linl.in_dim   # Number of polynomial terms
        n_rnl = rbasis_linl.out_dim    # Number of (n,l) basis functions

        # Extract polynomial basis (Chebyshev) from the WrappedBasis -> BranchLayer
        # rembed_layer.basis is WrappedBasis{BranchLayer{...}}
        # WrappedBasis has fields: l (the inner layer), len
        poly_basis = rembed_layer.basis.l.layers.layer_1  # The Chebyshev polynomial basis

        # Radial weights: ps.rembed.post.W[n_rnl, n_polys, n_species_pairs]
        W_radial = ps.rembed.post.W
    end

    # Readout weights: ps.readout.W[1, n_basis, n_species]
    W_readout = ps.readout.W
    n_basis = size(W_readout, 2)

    # Generate the export file
    open(filename, "w") do io
        _write_header(io, for_library)
        _write_species(io, _i2z)
        _write_tensor(io, tensor)

        # Write radial basis (hermite_spline or polynomial)
        if radial_basis == :hermite_spline
            @info "Extracting Hermite cubic spline data for exact trim-safe export..."
            # Extract the Hermite spline data from the already-splinified model
            # NOTE: The model should have been splinified BEFORE fitting, not here!
            # This function extracts the exact spline knots and coefficients.
            hermite_data = extract_hermite_spline_data(etace, ps, st, rcut)

            _write_spline_radial_basis_header(io, rcut)
            # Generate Hermite spline code using codegen (trim-safe, exact)
            spline_code = generate_hermite_spline_code(hermite_data, NZ, rcut)
            print(io, spline_code)
            println(io)
        elseif radial_basis == :polynomial
            _write_etace_radial_basis(io, etace, ps, agnesi_params, NZ, rcut)
        else
            error("Unknown radial_basis option: $radial_basis. Use :hermite_spline or :polynomial")
        end

        _write_spherical_harmonics(io, maxl)
        _write_etace_weights(io, W_readout, NZ, E0_dict, _i2z)
        _write_evaluation_functions(io, tensor, NZ, false)  # No pair potential in ETACE
        if for_library
            _write_c_interface(io, NZ)
        else
            _write_main(io, NZ)
        end
    end

    @info "Exported ETACE model to $filename (radial_basis=$radial_basis)"
    return filename
end

function _write_header(io, for_library::Bool)
    if for_library
        println(io, """
# ETACE Potential - Trim-compatible shared library export
# Generated by export_ace_model.jl
# Compile with: juliac --output-lib libace.so --trim=safe model.jl
#
# NOTE: This export is fully self-contained and trim=safe compatible.
# All spherical harmonics and radial basis evaluation code is inlined.
# No runtime dependency on SpheriCart, P4ML, or EquivariantTensors.

using StaticArrays
using StaticArrays: MVector
using LinearAlgebra: norm, dot

# ============================================================================
# MODEL CONSTANTS - Pre-computed from fitted ETACE model
# ============================================================================
""")
    else
        println(io, """
# ETACE Potential - Trim-compatible export
# Generated by export_ace_model.jl
# Compile with: juliac --output-exe model --trim=safe model.jl
#
# NOTE: This export is fully self-contained and trim=safe compatible.
# All spherical harmonics and radial basis evaluation code is inlined.
# No runtime dependency on SpheriCart, P4ML, or EquivariantTensors.

using StaticArrays
using StaticArrays: MVector
using LinearAlgebra: norm, dot

# ============================================================================
# MODEL CONSTANTS - Pre-computed from fitted ETACE model
# ============================================================================
""")
    end
end

function _write_species(io, _i2z)
    println(io, """
# Species mapping: index -> atomic number
const I2Z = $(_i2z)
const NZ = $(length(_i2z))

# Helper to convert atomic number to index
@inline function z2i(Z::Integer)
    @inbounds for i in 1:NZ
        I2Z[i] == Z && return i
    end
    error("Unknown atomic number: \$Z")
end
""")
end

function _write_spline_radial_basis_header(io, rcut)
    println(io, """
# ============================================================================
# RADIAL BASIS CONFIGURATION
# ============================================================================

const RCUT_MAX = $(rcut)

# Symmetric species pair indexing: (iz, jz) -> pair index
# For NZ=2: (1,1)->1, (1,2)->2, (2,2)->3
@inline function zz2pair_sym(iz::Int, jz::Int)::Int
    i, j = min(iz, jz), max(iz, jz)
    return (i - 1) * NZ - (i - 1) * (i - 2) ÷ 2 + (j - i + 1)
end
""")
end

function _write_tensor(io, tensor)
    # Extract specs
    abasis_spec = tensor.abasis.spec
    aabasis = tensor.aabasis

    # Write PooledSparseProduct spec (just the spec data, no ET dependency)
    println(io, "# PooledSparseProduct specification")
    println(io, "const ABASIS_SPEC = $(repr(collect(abasis_spec)))")
    println(io)

    # Write SparseSymmProd specs - as properly typed constants per order
    println(io, "# SparseSymmProd specification (typed per order)")
    for (ord, spec) in enumerate(aabasis.specs)
        if !isempty(spec)
            # Generate a tuple of tuples with explicit type
            tuples_str = join(["$(t)" for t in spec], ", ")
            println(io, "const AABASIS_SPECS_$(ord) = ($tuples_str,)  # Tuple of NTuple{$ord, Int}")
        end
    end
    println(io, "const AABASIS_RANGES = $(repr(aabasis.ranges))")
    println(io, "const AABASIS_HASCONST = $(aabasis.hasconst)")
    println(io)

    # Write A2B maps (just the sparse data, no SparseArrays dependency at runtime)
    println(io, "# A2B coupling matrices (sparse format: I, J, V)")
    for (idx, A2B) in enumerate(tensor.A2Bmaps)
        I, J, V = findnz(A2B)
        m, n = size(A2B)
        println(io, "const A2BMAP_$(idx)_I = $(repr(I))")
        println(io, "const A2BMAP_$(idx)_J = $(repr(J))")
        println(io, "const A2BMAP_$(idx)_V = $(repr(V))")
        println(io, "const A2BMAP_$(idx)_SIZE = ($m, $n)")
    end
    println(io)
end

function _write_etace_radial_basis(io, etace, ps, agnesi_params, NZ, rcut)
    println(io, """
# ============================================================================
# RADIAL BASIS (ETACE: Agnesi transform + polynomial + linear weights)
# ============================================================================
""")

    # Write cutoff
    println(io, "const RCUT_MAX = $(rcut)")
    println(io)

    # Note: Agnesi transform parameters are now written as individual constants
    # in _write_etace_radial_basis (per-pair specialized functions)

    # Write species pair index helper (symmetric)
    println(io, """
# Symmetric species pair indexing: (iz, jz) -> pair index
# For NZ=2: (1,1)->1, (1,2)->2, (2,2)->3
@inline function zz2pair_sym(iz::Int, jz::Int)::Int
    i, j = min(iz, jz), max(iz, jz)
    return (i - 1) * NZ - (i - 1) * (i - 2) ÷ 2 + (j - i + 1)
end
""")

    # Extract polynomial basis info
    # rembed_layer.basis is WrappedBasis{BranchLayer{layers=(layer_1=OrthPolyBasis, layer_2=...)}}
    # WrappedBasis has fields: l (the inner layer), len
    rembed_layer = etace.rembed.layer
    poly_basis = rembed_layer.basis.l.layers.layer_1
    n_polys = length(poly_basis)

    # Extract the actual polynomial coefficients (A, B, C for 3-term recurrence)
    # The ETACE uses an orthonormalized basis, not standard Chebyshev!
    poly_refstate = poly_basis.refstate
    poly_A = poly_refstate.A
    poly_B = poly_refstate.B
    poly_C = poly_refstate.C

    println(io, "# Polynomial basis (orthonormalized Chebyshev)")
    println(io, "# Uses 3-term recurrence: P[n+1] = (A[n]*y + B[n])*P[n] + C[n]*P[n-1]")
    println(io, "const N_POLYS = $(n_polys)")
    println(io, "const POLY_A = SVector{$(n_polys), Float64}($(repr(collect(poly_A))))")
    println(io, "const POLY_B = SVector{$(n_polys), Float64}($(repr(collect(poly_B))))")
    println(io, "const POLY_C = SVector{$(n_polys), Float64}($(repr(collect(poly_C))))")
    println(io)

    # Generate inline polynomial evaluation function (trim-safe, no P4ML dependency)
    println(io, """
# Inline polynomial evaluation (trim-safe, no P4ML dependency)
# 3-term recurrence: P[n+1] = (A[n]*y + B[n])*P[n] + C[n]*P[n-1]
@inline function eval_polys(y::T) where {T}
    P = MVector{N_POLYS, T}(undef)
    @inbounds begin
        P[1] = one(T)
        if N_POLYS >= 2
            P[2] = (POLY_A[1] * y + POLY_B[1]) * P[1]
        end
        for n = 2:N_POLYS-1
            P[n+1] = (POLY_A[n] * y + POLY_B[n]) * P[n] + POLY_C[n] * P[n-1]
        end
    end
    return P
end

# Inline polynomial evaluation with derivatives (trim-safe)
# d(P[n])/dy computed via differentiation of recurrence relation
@inline function eval_polys_ed(y::T) where {T}
    P = MVector{N_POLYS, T}(undef)
    dP = MVector{N_POLYS, T}(undef)
    @inbounds begin
        P[1] = one(T)
        dP[1] = zero(T)
        if N_POLYS >= 2
            P[2] = (POLY_A[1] * y + POLY_B[1]) * P[1]
            dP[2] = POLY_A[1] * P[1] + (POLY_A[1] * y + POLY_B[1]) * dP[1]
        end
        for n = 2:N_POLYS-1
            P[n+1] = (POLY_A[n] * y + POLY_B[n]) * P[n] + POLY_C[n] * P[n-1]
            dP[n+1] = POLY_A[n] * P[n] + (POLY_A[n] * y + POLY_B[n]) * dP[n] + POLY_C[n] * dP[n-1]
        end
    end
    return P, dP
end
""")

    # Write radial weights: W[n_rnl, n_polys, n_pairs]
    W_radial = ps.rembed.post.W
    n_rnl = size(W_radial, 1)
    println(io, "const N_RNL = $(n_rnl)")
    println(io)

    # Write weights per species pair
    n_pairs = size(W_radial, 3)
    println(io, "# Radial basis weights: W[n_rnl, n_polys] for each species pair")
    for pair_idx in 1:n_pairs
        W_pair = W_radial[:, :, pair_idx]
        println(io, "const RBASIS_W_$(pair_idx) = $(repr(collect(W_pair)))")
    end
    println(io)

    # Write Agnesi transform evaluation functions per pair
    # Trim-safe: all parameters are baked in as constants
    # This must match EquivariantTensors.eval_agnesi exactly:
    #   s = (r - rin) / (req - rin)
    #   x = 1 / (1 + a * s^pin / (1 + s^(pin - pcut)))
    #   y = clamp(b1 * x + b0, -1, 1)
    println(io, """
# Envelope: (1 - y^2)^2 for y ∈ [-1, 1]
@inline envelope(y::T) where {T} = (one(T) - y^2)^2
@inline envelope_d(y::T) where {T} = (one(T) - y^2)^2, -4 * y * (one(T) - y^2)
""")

    # Generate per-pair Agnesi functions with baked-in constants
    # Note: W_radial uses asymmetric indexing (NZ*NZ pairs), but Agnesi params use symmetric indexing
    # For NZ=2: asymmetric pairs (1,2,3,4) map to symmetric Agnesi params (1,2,2,3)
    # Asymmetric pair idx = (iz-1)*NZ + jz: (1,1)->1, (1,2)->2, (2,1)->3, (2,2)->4
    # Symmetric pair idx: (1,1)->1, (1,2)->2, (2,1)->2, (2,2)->3
    println(io, "# Per-pair Agnesi transforms (trim-safe: all parameters are constants)")
    for asym_pair_idx in 1:n_pairs
        # Convert asymmetric pair index to (iz, jz)
        iz = (asym_pair_idx - 1) ÷ NZ + 1
        jz = (asym_pair_idx - 1) % NZ + 1
        # Get symmetric pair index: min/max for symmetric indexing
        sym_i = min(iz, jz)
        sym_j = max(iz, jz)
        sym_pair_idx = (sym_i - 1) * NZ - (sym_i - 1) * (sym_i - 2) ÷ 2 + (sym_j - sym_i + 1)
        p = agnesi_params[sym_pair_idx]
        pair_idx = asym_pair_idx
        println(io, """
# Agnesi transform for pair $(pair_idx): r -> y ∈ [-1, 1]
const AGNESI_$(pair_idx)_RIN = $(Float64(p.rin))
const AGNESI_$(pair_idx)_REQ = $(Float64(p.req))
const AGNESI_$(pair_idx)_RCUT = $(Float64(rcut))
const AGNESI_$(pair_idx)_PIN = $(Float64(p.pin))
const AGNESI_$(pair_idx)_PCUT = $(Float64(p.pcut))
const AGNESI_$(pair_idx)_A = $(Float64(p.a))
const AGNESI_$(pair_idx)_B0 = $(Float64(p.b0))
const AGNESI_$(pair_idx)_B1 = $(Float64(p.b1))

@inline function eval_agnesi_$(pair_idx)(r::T) where {T}
    if r <= AGNESI_$(pair_idx)_RIN
        return one(T)
    end
    if r >= AGNESI_$(pair_idx)_RCUT
        return -one(T)
    end
    s = (r - AGNESI_$(pair_idx)_RIN) / (AGNESI_$(pair_idx)_REQ - AGNESI_$(pair_idx)_RIN)
    x = one(T) / (one(T) + AGNESI_$(pair_idx)_A * s^AGNESI_$(pair_idx)_PIN / (one(T) + s^(AGNESI_$(pair_idx)_PIN - AGNESI_$(pair_idx)_PCUT)))
    y = AGNESI_$(pair_idx)_B1 * x + AGNESI_$(pair_idx)_B0
    return clamp(y, -one(T), one(T))
end

@inline function eval_agnesi_d_$(pair_idx)(r::T) where {T}
    if r <= AGNESI_$(pair_idx)_RIN
        return one(T), zero(T)
    end
    if r >= AGNESI_$(pair_idx)_RCUT
        return -one(T), zero(T)
    end

    rin = AGNESI_$(pair_idx)_RIN
    req = AGNESI_$(pair_idx)_REQ
    pin = AGNESI_$(pair_idx)_PIN
    pcut = AGNESI_$(pair_idx)_PCUT
    a = AGNESI_$(pair_idx)_A
    b0 = AGNESI_$(pair_idx)_B0
    b1 = AGNESI_$(pair_idx)_B1

    s = (r - rin) / (req - rin)
    ds_dr = one(T) / (req - rin)

    # x = 1 / (1 + a * s^pin / (1 + s^(pin - pcut)))
    s_pin = s^pin
    s_diff = s^(pin - pcut)
    denom_inner = one(T) + s_diff
    denom = one(T) + a * s_pin / denom_inner
    x = one(T) / denom

    # y = b1 * x + b0
    y = b1 * x + b0

    # Derivative: dy/dr = b1 * dx/dr
    if s > 1e-10  # Avoid division by zero near s=0
        diff = pin - pcut
        ds_pin = pin * s^(pin - 1)
        ds_diff = diff > 0 ? diff * s^(diff - 1) : zero(T)
        d_denom_ds = a * (ds_pin * denom_inner - s_pin * ds_diff) / (denom_inner^2)
        dx_ds = -x^2 * d_denom_ds
        dy_ds = b1 * dx_ds
        dy_dr = dy_ds * ds_dr
    else
        dy_dr = zero(T)
    end

    if y <= -one(T) || y >= one(T)
        return clamp(y, -one(T), one(T)), zero(T)
    end
    return y, dy_dr
end
""")
    end

    # Generate per-pair specialized radial basis functions (trim-safe, better inlining)
    println(io, "# Per-pair specialized radial basis functions (trim-safe)")
    for pair_idx in 1:n_pairs
        println(io, """
@inline function evaluate_Rnl_$(pair_idx)(r::T)::SVector{N_RNL, T} where {T}
    # Transform distance to y ∈ [-1, 1] (using specialized Agnesi)
    y = eval_agnesi_$(pair_idx)(r)

    # Evaluate envelope
    env = envelope(y)
    if env <= zero(T)
        return zero(SVector{N_RNL, T})
    end

    # Evaluate polynomials at y (inline, trim-safe)
    P = eval_polys(y)

    # Apply envelope and linear layer
    return RBASIS_W_$(pair_idx) * SVector{N_POLYS, T}(env .* P)
end

@inline function evaluate_Rnl_d_$(pair_idx)(r::T)::Tuple{SVector{N_RNL, T}, SVector{N_RNL, T}} where {T}
    # Transform with derivative (using specialized Agnesi)
    y, dy_dr = eval_agnesi_d_$(pair_idx)(r)

    # Envelope with derivative
    env, denv_dy = envelope_d(y)
    denv_dr = denv_dy * dy_dr

    if env <= zero(T)
        return zero(SVector{N_RNL, T}), zero(SVector{N_RNL, T})
    end

    # Evaluate polynomials with derivatives (inline, trim-safe)
    P, dP = eval_polys_ed(y)
    dP_dr = dP .* dy_dr

    # P_env = env * P, d(P_env)/dr = denv/dr * P + env * dP/dr
    P_env = env .* P
    dP_env_dr = denv_dr .* P .+ env .* dP_dr

    # Linear layer
    Rnl = RBASIS_W_$(pair_idx) * SVector{N_POLYS, T}(P_env)
    dRnl = RBASIS_W_$(pair_idx) * SVector{N_POLYS, T}(dP_env_dr)

    return Rnl, dRnl
end
""")
    end

    # Write dispatcher functions
    println(io, """
# Radial basis dispatch: r, iz, jz -> Rnl vector
@inline function evaluate_Rnl(r::T, iz::Int, jz::Int)::SVector{N_RNL, T} where {T}
    pair_idx = zz2pair_sym(iz, jz)
""")
    for pair_idx in 1:n_pairs
        cond = pair_idx == 1 ? "if" : "elseif"
        println(io, "    $cond pair_idx == $pair_idx")
        println(io, "        return evaluate_Rnl_$(pair_idx)(r)")
    end
    println(io, "    end")
    println(io, "    return zero(SVector{N_RNL, T})  # fallback")
    println(io, "end")
    println(io)

    println(io, """
# Radial basis with derivatives dispatch
@inline function evaluate_Rnl_d(r::T, iz::Int, jz::Int)::Tuple{SVector{N_RNL, T}, SVector{N_RNL, T}} where {T}
    pair_idx = zz2pair_sym(iz, jz)
""")
    for pair_idx in 1:n_pairs
        cond = pair_idx == 1 ? "if" : "elseif"
        println(io, "    $cond pair_idx == $pair_idx")
        println(io, "        return evaluate_Rnl_d_$(pair_idx)(r)")
    end
    println(io, "    end")
    println(io, "    return zero(SVector{N_RNL, T}), zero(SVector{N_RNL, T})  # fallback")
    println(io, "end")
end

# Keep the old function for backward compatibility error message
function _write_radial_basis(io, rbasis::ACEpotentials.Models.SplineRnlrzzBasis, NZ)
    println(io, """
# ============================================================================
# RADIAL BASIS (Spline-based)
# ============================================================================
""")

    # Write rin0cuts
    println(io, "# Cutoff parameters: (rin, r0, rcut) for each species pair")
    rcut_max = 0.0
    for iz in 1:NZ
        for jz in 1:NZ
            rin0cut = rbasis.rin0cuts[iz, jz]
            println(io, "const RIN0CUT_$(iz)_$(jz) = (rin=$(rin0cut.rin), r0=$(rin0cut.r0), rcut=$(rin0cut.rcut))")
            rcut_max = max(rcut_max, rin0cut.rcut)
        end
    end
    # Maximum cutoff across all species pairs (used for neighbor list construction)
    println(io, "const RCUT_MAX = $(rcut_max)")
    println(io)

    # Write spec
    println(io, "const R_SPEC = $(repr(rbasis.spec))")
    println(io, "const N_RNL = $(length(rbasis.spec))")
    println(io)

    # Write transforms
    println(io, "# Radial transforms")
    for iz in 1:NZ
        for jz in 1:NZ
            trans = rbasis.transforms[iz, jz]
            _write_transform(io, trans, iz, jz)
        end
    end

    # Write envelopes
    println(io, "# Radial envelopes")
    for iz in 1:NZ
        for jz in 1:NZ
            env = rbasis.envelopes[iz, jz]
            _write_envelope(io, env, iz, jz)
        end
    end

    # Write splines - this is the key data for evaluation
    println(io, "# Spline knots and coefficients")
    for iz in 1:NZ
        for jz in 1:NZ
            spl = rbasis.splines[iz, jz]
            _write_spline(io, spl, iz, jz)
        end
    end

    # Write evaluation function
    println(io, """

# Radial basis evaluation (returns SVector{N_RNL, T})
@inline function evaluate_Rnl(r::T, iz::Int, jz::Int)::SVector{N_RNL, T} where {T}
    # Get transform and envelope for this species pair
    # (Dispatched at compile time for small NZ)
    """)

    for iz in 1:NZ
        for jz in 1:NZ
            cond = iz == 1 && jz == 1 ? "if" : "elseif"
            println(io, "    $cond iz == $iz && jz == $jz")
            println(io, "        rcut = RIN0CUT_$(iz)_$(jz).rcut")
            println(io, "        if r >= rcut; return zero(SVector{N_RNL, T}); end")
            println(io, "        x = transform_$(iz)_$(jz)(r)")
            println(io, "        env = envelope_$(iz)_$(jz)(r, x)")
            println(io, "        return env .* spline_$(iz)_$(jz)(x)")
        end
    end
    println(io, "    end")
    println(io, "    # Fallback for unknown species pair")
    println(io, "    return zero(SVector{N_RNL, T})")
    println(io, "end")
    println(io)

    # Write evaluation with derivative function
    println(io, """
# Radial basis evaluation with derivative dRnl/dr
@inline function evaluate_Rnl_d(r::T, iz::Int, jz::Int)::Tuple{SVector{N_RNL, T}, SVector{N_RNL, T}} where {T}
    """)

    for iz in 1:NZ
        for jz in 1:NZ
            cond = iz == 1 && jz == 1 ? "if" : "elseif"
            println(io, "    $cond iz == $iz && jz == $jz")
            println(io, "        rcut = RIN0CUT_$(iz)_$(jz).rcut")
            println(io, "        if r >= rcut; return zero(SVector{N_RNL, T}), zero(SVector{N_RNL, T}); end")
            println(io, "        # Transform with derivative")
            println(io, "        x, dx_dr = transform_d_$(iz)_$(jz)(r)")
            println(io, "        # Envelope with derivative")
            println(io, "        env, denv_dx = envelope_d_$(iz)_$(jz)(r, x)")
            println(io, "        # Spline with derivative")
            println(io, "        spl, dspl_dx = spline_d_$(iz)_$(jz)(x)")
            println(io, "        # Rnl = env * spl")
            println(io, "        Rnl = env .* spl")
            println(io, "        # dRnl/dr = denv/dr * spl + env * dspl/dr")
            println(io, "        #         = denv/dx * dx/dr * spl + env * dspl/dx * dx/dr")
            println(io, "        dRnl_dr = (denv_dx * dx_dr) .* spl .+ env .* dspl_dx .* dx_dr")
            println(io, "        return Rnl, dRnl_dr")
        end
    end
    println(io, "    end")
    println(io, "    # Fallback for unknown species pair")
    println(io, "    return zero(SVector{N_RNL, T}), zero(SVector{N_RNL, T})")
    println(io, "end")
    println(io)
end

function _write_transform(io, trans::ACEpotentials.Models.NormalizedTransform, iz, jz)
    inner = trans.trans
    println(io, """
# Transform $(iz)-$(jz): NormalizedTransform
const TRANS_$(iz)_$(jz)_YIN = $(trans.yin)
const TRANS_$(iz)_$(jz)_YCUT = $(trans.ycut)
const TRANS_$(iz)_$(jz)_P = $(inner.p)
const TRANS_$(iz)_$(jz)_Q = $(inner.q)
const TRANS_$(iz)_$(jz)_R0 = $(inner.r0)
const TRANS_$(iz)_$(jz)_RIN = $(inner.rin)
const TRANS_$(iz)_$(jz)_A = $(inner.a)

@inline function transform_$(iz)_$(jz)(r::T) where {T}
    rin = TRANS_$(iz)_$(jz)_RIN
    if r <= rin
        return one(T)
    end
    r0, q, p, a = TRANS_$(iz)_$(jz)_R0, TRANS_$(iz)_$(jz)_Q, TRANS_$(iz)_$(jz)_P, TRANS_$(iz)_$(jz)_A
    s = (r - rin) / (r0 - rin)
    y = (a + s^q) / (a + s^p)
    # Normalize
    yin, ycut = TRANS_$(iz)_$(jz)_YIN, TRANS_$(iz)_$(jz)_YCUT
    return clamp(-one(T) + 2 * (y - yin) / (ycut - yin), -one(T), one(T))
end

# Transform with derivative dx/dr
@inline function transform_d_$(iz)_$(jz)(r::T) where {T}
    rin = TRANS_$(iz)_$(jz)_RIN
    if r <= rin
        return one(T), zero(T)
    end
    r0, q, p, a = TRANS_$(iz)_$(jz)_R0, TRANS_$(iz)_$(jz)_Q, TRANS_$(iz)_$(jz)_P, TRANS_$(iz)_$(jz)_A
    yin, ycut = TRANS_$(iz)_$(jz)_YIN, TRANS_$(iz)_$(jz)_YCUT

    s = (r - rin) / (r0 - rin)
    ds_dr = one(T) / (r0 - rin)

    # y = (a + s^q) / (a + s^p)
    num = a + s^q
    den = a + s^p
    y = num / den

    # dy/ds = (q*s^(q-1)*den - p*s^(p-1)*num) / den^2
    dnum_ds = q > 0 ? q * s^(q-1) : zero(T)
    dden_ds = p > 0 ? p * s^(p-1) : zero(T)
    dy_ds = (dnum_ds * den - dden_ds * num) / (den * den)
    dy_dr = dy_ds * ds_dr

    # x = -1 + 2*(y - yin)/(ycut - yin)
    scale = 2 / (ycut - yin)
    x = -one(T) + scale * (y - yin)
    dx_dr = scale * dy_dr

    # Clamp: if outside [-1, 1], derivative is zero
    if x <= -one(T) || x >= one(T)
        return clamp(x, -one(T), one(T)), zero(T)
    end
    return x, dx_dr
end
""")
end

function _write_envelope(io, env::ACEpotentials.Models.PolyEnvelope2sX, iz, jz)
    println(io, """
# Envelope $(iz)-$(jz): PolyEnvelope2sX
const ENV_$(iz)_$(jz)_X1 = $(env.x1)
const ENV_$(iz)_$(jz)_X2 = $(env.x2)
const ENV_$(iz)_$(jz)_P1 = $(env.p1)
const ENV_$(iz)_$(jz)_P2 = $(env.p2)
const ENV_$(iz)_$(jz)_S = $(env.s)

@inline function envelope_$(iz)_$(jz)(r::T, x::T) where {T}
    x1, x2 = ENV_$(iz)_$(jz)_X1, ENV_$(iz)_$(jz)_X2
    if !(x1 < x < x2)
        return zero(T)
    end
    p1, p2, s = ENV_$(iz)_$(jz)_P1, ENV_$(iz)_$(jz)_P2, ENV_$(iz)_$(jz)_S
    return s * (x - x1)^p1 * (x2 - x)^p2
end

# Envelope with derivative denv/dx
@inline function envelope_d_$(iz)_$(jz)(r::T, x::T) where {T}
    x1, x2 = ENV_$(iz)_$(jz)_X1, ENV_$(iz)_$(jz)_X2
    if !(x1 < x < x2)
        return zero(T), zero(T)
    end
    p1, p2, s = ENV_$(iz)_$(jz)_P1, ENV_$(iz)_$(jz)_P2, ENV_$(iz)_$(jz)_S

    # env = s * (x - x1)^p1 * (x2 - x)^p2
    left = (x - x1)^p1
    right = (x2 - x)^p2
    env = s * left * right

    # denv/dx = s * (p1*(x-x1)^(p1-1)*(x2-x)^p2 - p2*(x-x1)^p1*(x2-x)^(p2-1))
    dleft_dx = p1 > 0 ? p1 * (x - x1)^(p1-1) : zero(T)
    dright_dx = p2 > 0 ? -p2 * (x2 - x)^(p2-1) : zero(T)
    denv_dx = s * (dleft_dx * right + left * dright_dx)

    return env, denv_dx
end
""")
end

function _write_spline(io, spl, iz, jz)
    # Extract spline data
    # The spline is a ScaledInterpolation wrapping a BSplineInterpolation
    itp = spl.itp  # ScaledInterpolation
    inner_itp = itp.itp  # BSplineInterpolation

    # Get the knots and coefficients
    knots = itp.ranges[1]  # The x values (StepRangeLen)
    x_start = first(knots)
    x_step = step(knots)
    x_len = length(knots)

    # Get coefficients (these are SVector values)
    coeffs = inner_itp.coefs
    n_rnl = length(first(coeffs))

    # Write coefficients as a Tuple of SVectors for type stability
    println(io, """
# Spline $(iz)-$(jz)
const SPL_$(iz)_$(jz)_XSTART = $(x_start)
const SPL_$(iz)_$(jz)_XSTEP = $(x_step)
const SPL_$(iz)_$(jz)_XLEN = $(x_len)
""")

    # Write each coefficient as an SVector
    print(io, "const SPL_$(iz)_$(jz)_COEFFS = (\n")
    for (i, c) in enumerate(coeffs)
        vals = join(["$v" for v in c], ", ")
        print(io, "    SVector{$(n_rnl), Float64}($(vals))")
        if i < length(coeffs)
            println(io, ",")
        else
            println(io)
        end
    end
    println(io, ")")
    println(io)

    println(io, """
# Linear interpolation of spline coefficients
@inline function spline_$(iz)_$(jz)(x::T) where {T}
    # Find interval
    idx_f = (x - SPL_$(iz)_$(jz)_XSTART) / SPL_$(iz)_$(jz)_XSTEP + 1
    idx = floor(Int, idx_f)
    idx = clamp(idx, 1, SPL_$(iz)_$(jz)_XLEN - 1)
    t = T(idx_f - idx)

    # Linear interpolation between spline coefficients (type-stable via Tuple indexing)
    @inbounds c0 = SPL_$(iz)_$(jz)_COEFFS[idx]
    @inbounds c1 = SPL_$(iz)_$(jz)_COEFFS[idx+1]
    return (one(T) - t) .* c0 .+ t .* c1
end

# Spline with derivative dspl/dx
@inline function spline_d_$(iz)_$(jz)(x::T) where {T}
    # Find interval
    idx_f = (x - SPL_$(iz)_$(jz)_XSTART) / SPL_$(iz)_$(jz)_XSTEP + 1
    idx = floor(Int, idx_f)
    idx = clamp(idx, 1, SPL_$(iz)_$(jz)_XLEN - 1)
    t = T(idx_f - idx)

    @inbounds c0 = SPL_$(iz)_$(jz)_COEFFS[idx]
    @inbounds c1 = SPL_$(iz)_$(jz)_COEFFS[idx+1]

    # spl = (1-t)*c0 + t*c1
    spl = (one(T) - t) .* c0 .+ t .* c1

    # dspl/dx = dspl/dt * dt/dx = (c1 - c0) / step
    dspl_dx = (c1 .- c0) ./ T(SPL_$(iz)_$(jz)_XSTEP)

    return spl, dspl_dx
end
""")
end

function _write_spherical_harmonics(io, maxl)
    # Generate inline solid harmonics code using SpheriCart's code generators
    # This produces trim-safe code that doesn't require SpheriCart at runtime
    ylm_code = generate_solid_harmonics_code(maxl)

    # Write the generated code
    print(io, ylm_code)
    println(io)
end

function _write_etace_weights(io, W_readout, NZ, E0_dict::Union{Dict{Int,Float64},Nothing}, _i2z)
    println(io, """
# ============================================================================
# MODEL WEIGHTS (ETACE: readout weights only, no pair potential)
# ============================================================================
""")

    # W_readout has shape [1, n_basis, n_species]
    n_basis = size(W_readout, 2)
    println(io, "# Number of basis functions")
    println(io, "const N_BASIS = $(n_basis)")
    println(io)

    # Write WB weights per species
    println(io, "# B basis weights (per species)")
    for iz in 1:NZ
        w = W_readout[1, :, iz]
        println(io, "const WB_$(iz) = $(repr(collect(w)))")
    end
    println(io)

    # ETACE has no pair potential
    println(io, "# Note: ETACE has no pair potential (many-body only)")
    println(io)

    # Write E0 values
    println(io, """
# ============================================================================
# REFERENCE ENERGIES (E0)
# ============================================================================
""")

    if E0_dict !== nothing
        println(io, "# E0 values extracted from ETOneBody calculator")
        for (iz, Z) in enumerate(_i2z)
            E0 = get(E0_dict, Z, 0.0)
            println(io, "const E0_$(iz) = $(E0)  # Z=$(Z)")
        end
    else
        println(io, "# Note: ETACE calculates many-body energy only. E0 contributions should be")
        println(io, "# added separately using StackedCalculator with ETOneBody.")
        for iz in 1:NZ
            Z = _i2z[iz]
            println(io, "const E0_$(iz) = 0.0  # Z=$(Z)")
        end
    end
    println(io)
end

function _write_weights(io, WB, Wpair, NZ)
    println(io, """
# ============================================================================
# MODEL WEIGHTS
# ============================================================================
""")

    # Write basis size constant
    n_basis = size(WB, 1)
    println(io, "# Number of basis functions")
    println(io, "const N_BASIS = $(n_basis)")
    println(io)

    # Write WB weights
    println(io, "# B basis weights (per species)")
    for iz in 1:NZ
        w = WB[:, iz]
        println(io, "const WB_$(iz) = $(repr(collect(w)))")
    end
    println(io)

    # Write Wpair weights if present
    if Wpair !== nothing
        println(io, "# Pair basis weights (per species)")
        for iz in 1:NZ
            w = Wpair[:, iz]
            println(io, "const WPAIR_$(iz) = $(repr(collect(w)))")
        end
        println(io)
    end
end

function _write_vref(io, Vref, _i2z)
    println(io, """
# ============================================================================
# REFERENCE ENERGIES (E0)
# ============================================================================
""")

    # Extract E0 values
    for (iz, Z) in enumerate(_i2z)
        E0 = Vref.E0[Z]
        println(io, "const E0_$(iz) = $(E0)  # Z=$(Z)")
    end
    println(io)
end

function _write_evaluation_functions(io, tensor, NZ, has_pair)
    # Get dimensions for pre-allocation
    nA = length(tensor.abasis)
    nAA = length(tensor.aabasis)

    println(io, """
# ============================================================================
# PRE-ALLOCATED WORK ARRAYS (avoid allocations in hot paths)
# ============================================================================

const MAX_NEIGHBORS = 256  # Maximum number of neighbors per site

# Work arrays for embeddings (indexed as [neighbor, feature])
# Layout matches the abasis inner loop: Rnl[j, ϕ1] with j varying is contiguous
const WORK_Rnl = zeros(Float64, MAX_NEIGHBORS, N_RNL)
const WORK_dRnl = zeros(Float64, MAX_NEIGHBORS, N_RNL)
const WORK_Ylm = zeros(Float64, MAX_NEIGHBORS, N_YLM)
const WORK_dYlm = zeros(SVector{3, Float64}, MAX_NEIGHBORS, N_YLM)
const WORK_rs = zeros(Float64, MAX_NEIGHBORS)
const WORK_rhats = zeros(SVector{3, Float64}, MAX_NEIGHBORS)

# Work arrays for tensor evaluation
const WORK_A = zeros(Float64, $nA)
const WORK_AA = zeros(Float64, $nAA)
const WORK_B = zeros(Float64, N_BASIS)

# Work arrays for pullback
const WORK_∂A = zeros(Float64, $nA)
const WORK_∂AA = zeros(Float64, $nAA)
const WORK_∂Rnl = zeros(Float64, MAX_NEIGHBORS, N_RNL)
const WORK_∂Ylm = zeros(Float64, MAX_NEIGHBORS, N_YLM)
const WORK_∂B = zeros(Float64, N_BASIS)

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

# Compute embeddings for all neighbors (values only)
# Uses pre-allocated arrays, returns views
function compute_embeddings(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    nneigh = length(Rs)
    @assert nneigh <= MAX_NEIGHBORS "Too many neighbors: \$nneigh > \$MAX_NEIGHBORS"
    iz0 = z2i(Z0)

    # Get views into pre-allocated arrays
    Rnl = view(WORK_Rnl, 1:nneigh, :)
    Ylm = view(WORK_Ylm, 1:nneigh, :)

    @inbounds for j in 1:nneigh
        r = norm(Rs[j])
        if r > 1e-10
            jz = z2i(Zs[j])
            Rnl_j = evaluate_Rnl(r, iz0, jz)
            @simd for t in 1:N_RNL
                WORK_Rnl[j, t] = Rnl_j[t]
            end

            # Solid harmonics: evaluate with full R vector (not unit vector!)
            Ylm_j = eval_ylm(Rs[j])
            @simd for t in 1:N_YLM
                WORK_Ylm[j, t] = Ylm_j[t]
            end
        else
            # Zero out for this neighbor (r too small)
            @simd for t in 1:N_RNL
                WORK_Rnl[j, t] = 0.0
            end
            @simd for t in 1:N_YLM
                WORK_Ylm[j, t] = 0.0
            end
        end
    end

    return Rnl, Ylm
end

# Compute embeddings with derivatives for analytic forces
# Uses pre-allocated arrays, returns views
function compute_embeddings_ed(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    nneigh = length(Rs)
    @assert nneigh <= MAX_NEIGHBORS "Too many neighbors: \$nneigh > \$MAX_NEIGHBORS"
    iz0 = z2i(Z0)

    # Get views into pre-allocated arrays
    Rnl = view(WORK_Rnl, 1:nneigh, :)
    dRnl = view(WORK_dRnl, 1:nneigh, :)
    Ylm = view(WORK_Ylm, 1:nneigh, :)
    dYlm = view(WORK_dYlm, 1:nneigh, :)
    rs = view(WORK_rs, 1:nneigh)
    rhats = view(WORK_rhats, 1:nneigh)

    @inbounds for j in 1:nneigh
        r = norm(Rs[j])
        WORK_rs[j] = r
        if r > 1e-10
            rhat = Rs[j] / r
            WORK_rhats[j] = rhat
            jz = z2i(Zs[j])

            # Radial basis with derivative
            Rnl_j, dRnl_j = evaluate_Rnl_d(r, iz0, jz)
            @simd for t in 1:N_RNL
                WORK_Rnl[j, t] = Rnl_j[t]
                WORK_dRnl[j, t] = dRnl_j[t]
            end

            # Solid harmonics with derivatives (evaluated with full R vector)
            # dYlm returns dR_lm/dR directly (not dY_lm/dr̂)
            Ylm_j, dYlm_j = eval_ylm_ed(Rs[j])
            @simd for t in 1:N_YLM
                WORK_Ylm[j, t] = Ylm_j[t]
                WORK_dYlm[j, t] = dYlm_j[t]
            end
        else
            # Zero out for this neighbor (r too small)
            WORK_rhats[j] = zero(SVector{3, Float64})
            @simd for t in 1:N_RNL
                WORK_Rnl[j, t] = 0.0
                WORK_dRnl[j, t] = 0.0
            end
            @simd for t in 1:N_YLM
                WORK_Ylm[j, t] = 0.0
                WORK_dYlm[j, t] = zero(SVector{3, Float64})
            end
        end
    end

    return Rnl, dRnl, Ylm, dYlm, rs, rhats
end

# Site energy evaluation
function site_energy(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    iz0 = z2i(Z0)

    if length(Rs) == 0
        # Return E0 for isolated atom
""")

    # Write E0 lookup
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "        $cond iz0 == $iz; return E0_$iz")
    end
    println(io, "        end")

    println(io, """
    end

    # Compute embeddings
    Rnl, Ylm = compute_embeddings(Rs, Zs, Z0)

    # Evaluate tensor (using manual inline evaluation, trim-safe)
    B, _ = tensor_evaluate(Rnl, Ylm)

    # Contract with weights
    val = 0.0
""")

    # Write weight contraction
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "    $cond iz0 == $iz")
        println(io, "        val = dot(B, WB_$iz)")
    end
    println(io, "    end")

    # Add E0
    println(io, """

    # Add reference energy
""")
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "    $cond iz0 == $iz; val += E0_$iz")
    end
    println(io, "    end")

    println(io, """

    return val
end

# Site basis evaluation (returns raw basis vector without weight contraction)
function site_basis(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    if length(Rs) == 0
        # Return zeros for isolated atom
        return zeros(Float64, N_BASIS)
    end

    # Compute embeddings
    Rnl, Ylm = compute_embeddings(Rs, Zs, Z0)

    # Evaluate tensor (using manual inline evaluation, trim-safe)
    B, _ = tensor_evaluate(Rnl, Ylm)

    return collect(B)
end

# ============================================================================
# MANUAL PULLBACK FUNCTIONS (trim-safe)
# ============================================================================

# Static product with gradient (for SparseSymmProd pullback)
@inline function _static_prod_ed(b::NTuple{1, T}) where {T}
    return b[1], (one(T),)
end

@inline function _static_prod_ed(b::NTuple{2, T}) where {T}
    return b[1] * b[2], (b[2], b[1])
end

@inline function _static_prod_ed(b::NTuple{3, T}) where {T}
    p12 = b[1] * b[2]
    return p12 * b[3], (b[2] * b[3], b[1] * b[3], p12)
end

@inline function _static_prod_ed(b::NTuple{4, T}) where {T}
    p12 = b[1] * b[2]
    p34 = b[3] * b[4]
    return p12 * p34, (b[2] * p34, b[1] * p34, p12 * b[4], p12 * b[3])
end

# Manual pullback through PooledSparseProduct (abasis): ∂A -> (∂Rnl, ∂Ylm)
# This is the key function that replaces ET.pullback for the abasis
@inline function pullback_abasis!(∂Rnl, ∂Ylm, ∂A, Rnl, Ylm)
    nX = size(Rnl, 1)

    @inbounds for (iA, ϕ) in enumerate(ABASIS_SPEC)
        ϕ1, ϕ2 = ϕ  # (Rnl index, Ylm index)
        ∂A_iA = ∂A[iA]
        @simd ivdep for j = 1:nX
            ∂Rnl[j, ϕ1] += ∂A_iA * Ylm[j, ϕ2]
            ∂Ylm[j, ϕ2] += ∂A_iA * Rnl[j, ϕ1]
        end
    end
    return ∂Rnl, ∂Ylm
end
""")

    # Write the aabasis pullback with the correct order
    aabasis = tensor.aabasis
    max_order = length(aabasis.specs)

    println(io, """
# Manual pullback through SparseSymmProd (aabasis): ∂AA -> ∂A
# Note: Accepts any array types to support views
@inline function pullback_aabasis!(∂A, ∂AA, A)
""")

    # Generate pullback code for each order
    for ord in 1:max_order
        spec = aabasis.specs[ord]
        range_start = aabasis.ranges[ord].start
        range_stop = aabasis.ranges[ord].stop

        if isempty(spec)
            continue
        end

        println(io, "    # Order $ord terms (indices $range_start:$range_stop)")
        println(io, "    @inbounds for (i_local, ϕ) in enumerate(AABASIS_SPECS_$ord)")
        println(io, "        i = $(range_start - 1) + i_local")
        println(io, "        ∂AA_i = ∂AA[i]")

        if ord == 1
            println(io, "        ∂A[ϕ[1]] += ∂AA_i")
        elseif ord == 2
            println(io, "        a1, a2 = A[ϕ[1]], A[ϕ[2]]")
            println(io, "        ∂A[ϕ[1]] += ∂AA_i * a2")
            println(io, "        ∂A[ϕ[2]] += ∂AA_i * a1")
        elseif ord == 3
            println(io, "        a1, a2, a3 = A[ϕ[1]], A[ϕ[2]], A[ϕ[3]]")
            println(io, "        ∂A[ϕ[1]] += ∂AA_i * a2 * a3")
            println(io, "        ∂A[ϕ[2]] += ∂AA_i * a1 * a3")
            println(io, "        ∂A[ϕ[3]] += ∂AA_i * a1 * a2")
        elseif ord == 4
            println(io, "        a1, a2, a3, a4 = A[ϕ[1]], A[ϕ[2]], A[ϕ[3]], A[ϕ[4]]")
            println(io, "        ∂A[ϕ[1]] += ∂AA_i * a2 * a3 * a4")
            println(io, "        ∂A[ϕ[2]] += ∂AA_i * a1 * a3 * a4")
            println(io, "        ∂A[ϕ[3]] += ∂AA_i * a1 * a2 * a4")
            println(io, "        ∂A[ϕ[4]] += ∂AA_i * a1 * a2 * a3")
        else
            # General case using _static_prod_ed
            println(io, "        aa = ntuple(t -> A[ϕ[t]], Val($ord))")
            println(io, "        _, gi = _static_prod_ed(aa)")
            println(io, "        for t in 1:$ord")
            println(io, "            ∂A[ϕ[t]] += ∂AA_i * gi[t]")
            println(io, "        end")
        end
        println(io, "    end")
        println(io)
    end

    println(io, "    return ∂A")
    println(io, "end")
    println(io)

    # Write the full tensor pullback
    println(io, """
# ============================================================================
# MANUAL FORWARD EVALUATION FUNCTIONS (trim-safe)
# ============================================================================

# Manual forward pass through PooledSparseProduct (abasis): (Rnl, Ylm) -> A
# This replaces ET.evaluate! for the abasis
@inline function evaluate_abasis!(A, Rnl, Ylm)
    nX = size(Rnl, 1)

    @inbounds for (iA, ϕ) in enumerate(ABASIS_SPEC)
        ϕ1, ϕ2 = ϕ  # (Rnl index, Ylm index)
        acc = 0.0
        @simd ivdep for j = 1:nX
            acc += Rnl[j, ϕ1] * Ylm[j, ϕ2]
        end
        A[iA] = acc
    end
    return A
end""")

    # Get number of A basis functions
    nA = length(tensor.abasis)
    nAA = length(tensor.aabasis)

    println(io, """

# Manual forward pass through SparseSymmProd (aabasis): A -> AA
# This replaces ET.evaluate for the aabasis
# Note: Accepts any array types to support views
@inline function evaluate_aabasis!(AA, A)
""")

    # Generate forward pass code for each order
    aabasis = tensor.aabasis
    max_order = length(aabasis.specs)

    for ord in 1:max_order
        spec = aabasis.specs[ord]
        range_start = aabasis.ranges[ord].start
        range_stop = aabasis.ranges[ord].stop

        if isempty(spec)
            continue
        end

        println(io, "    # Order $ord terms (indices $range_start:$range_stop)")
        println(io, "    @inbounds for (i_local, ϕ) in enumerate(AABASIS_SPECS_$ord)")
        println(io, "        i = $(range_start - 1) + i_local")

        if ord == 1
            println(io, "        AA[i] = A[ϕ[1]]")
        elseif ord == 2
            println(io, "        AA[i] = A[ϕ[1]] * A[ϕ[2]]")
        elseif ord == 3
            println(io, "        AA[i] = A[ϕ[1]] * A[ϕ[2]] * A[ϕ[3]]")
        elseif ord == 4
            println(io, "        AA[i] = A[ϕ[1]] * A[ϕ[2]] * A[ϕ[3]] * A[ϕ[4]]")
        else
            # General case
            println(io, "        AA[i] = prod(A[ϕ[t]] for t in 1:$ord)")
        end
        println(io, "    end")
        println(io)
    end

    println(io, "    return AA")
    println(io, "end")
    println(io)

    # Write full tensor forward evaluation
    println(io, """
# Full manual forward pass through tensor: (Rnl, Ylm) -> B
# This replaces ET.evaluate with a trim-safe implementation
# Uses pre-allocated work arrays to avoid allocations
function tensor_evaluate(Rnl, Ylm)
    # Reset and use pre-allocated arrays
    fill!(WORK_A, 0.0)
    fill!(WORK_AA, 0.0)
    fill!(WORK_B, 0.0)

    # Step 1: A = evaluate(abasis, Rnl, Ylm)
    evaluate_abasis!(WORK_A, Rnl, Ylm)

    # Step 2: AA = evaluate(aabasis, A)
    evaluate_aabasis!(WORK_AA, WORK_A)

    # Step 3: B = A2Bmap * AA (sparse matrix-vector multiplication)
    @inbounds for (idx, I) in enumerate(A2BMAP_1_I)
        J = A2BMAP_1_J[idx]
        V = A2BMAP_1_V[idx]
        WORK_B[I] += V * WORK_AA[J]
    end

    return WORK_B, WORK_A
end

# ============================================================================
# MANUAL PULLBACK FUNCTIONS (trim-safe)
# ============================================================================

# Full manual pullback through tensor: ∂B -> (∂Rnl, ∂Ylm)
# This replaces ET.pullback with a trim-safe implementation
# Uses pre-allocated work arrays to avoid allocations
function tensor_pullback!(∂Rnl, ∂Ylm, ∂B, Rnl, Ylm, A)
    # Reset pre-allocated arrays
    fill!(WORK_∂AA, 0.0)
    fill!(WORK_∂A, 0.0)

    # Step 1: ∂AA = A2Bmap' * ∂B (sparse transpose multiplication)
    @inbounds for (I_idx, I) in enumerate(A2BMAP_1_I)
        J = A2BMAP_1_J[I_idx]
        V = A2BMAP_1_V[I_idx]
        WORK_∂AA[J] += V * ∂B[I]  # Transpose: A2Bmap'[J,I] = A2Bmap[I,J]
    end

    # Step 2: ∂A = pullback_aabasis(∂AA, A)
    pullback_aabasis!(WORK_∂A, WORK_∂AA, A)

    # Step 3: (∂Rnl, ∂Ylm) = pullback_abasis(∂A, Rnl, Ylm)
    pullback_abasis!(∂Rnl, ∂Ylm, WORK_∂A, Rnl, Ylm)

    return ∂Rnl, ∂Ylm
end

# Site energy with ANALYTIC forces using manual pullback (trim-safe)
function site_energy_forces(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    iz0 = z2i(Z0)
    nneigh = length(Rs)

    if nneigh == 0
        # Return E0 for isolated atom, no forces
""")

    # Write E0 lookup for forces function
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "        $cond iz0 == $iz; return E0_$iz, SVector{3, Float64}[]")
    end
    println(io, "        end")

    println(io, """
    end

    # Compute embeddings with derivatives
    Rnl, dRnl, Ylm, dYlm, rs, rhats = compute_embeddings_ed(Rs, Zs, Z0)

    # Evaluate tensor (using manual inline evaluation, trim-safe)
    # Returns both B and A (A needed for pullback)
    B, A = tensor_evaluate(Rnl, Ylm)

    # Contract with weights to get energy
    # Use pre-allocated ∂B array
    fill!(WORK_∂B, 0.0)
    Ei = 0.0
""")

    # Write weight contraction for forces
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "    $cond iz0 == $iz")
        println(io, "        Ei = dot(B, WB_$iz)")
        println(io, "        for k in 1:N_BASIS; WORK_∂B[k] = WB_$(iz)[k]; end  # ∂Ei/∂B = WB")
    end
    println(io, "    end")

    println(io, """

    # Backward pass through tensor using MANUAL pullback (trim-safe)
    # Use pre-allocated arrays for gradients
    ∂Rnl = view(WORK_∂Rnl, 1:nneigh, :)
    ∂Ylm = view(WORK_∂Ylm, 1:nneigh, :)
    @inbounds for j in 1:nneigh
        @simd for t in 1:N_RNL; WORK_∂Rnl[j, t] = 0.0; end
        @simd for t in 1:N_YLM; WORK_∂Ylm[j, t] = 0.0; end
    end
    tensor_pullback!(∂Rnl, ∂Ylm, WORK_∂B, Rnl, Ylm, A)

    # Assemble forces: ∂Ei/∂Rⱼ
    forces = Vector{SVector{3, Float64}}(undef, nneigh)
    @inbounds for j in 1:nneigh
        f = zero(SVector{3, Float64})
        r = rs[j]
        if r > 1e-10
            rhat = rhats[j]
            # Contribution from radial basis: ∂Ei/∂Rnl * dRnl/dr * r̂
            for t in 1:N_RNL
                f = f + (∂Rnl[j, t] * dRnl[j, t]) * rhat
            end
            # Contribution from solid harmonics: ∂Ei/∂R_lm * dR_lm/dR
            # (dYlm is dR_lm/dR for solid harmonics - direct gradient, no chain rule needed)
            for t in 1:N_YLM
                f = f + ∂Ylm[j, t] * dYlm[j, t]
            end
        end
        forces[j] = -f  # Force is negative gradient
    end

    # Add reference energy
""")

    # Add E0
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "    $cond iz0 == $iz; Ei += E0_$iz")
    end
    println(io, "    end")

    println(io, """

    return Ei, forces
end

# Site energy with forces AND virial stress using manual pullback (trim-safe)
function site_energy_forces_virial(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    iz0 = z2i(Z0)
    nneigh = length(Rs)

    # Initialize virial tensor (3x3 symmetric)
    virial = zeros(SMatrix{3, 3, Float64, 9})

    if nneigh == 0
        # Return E0 for isolated atom, no forces/virial
""")

    # Write E0 lookup for virial function
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "        $cond iz0 == $iz; return E0_$iz, SVector{3, Float64}[], virial")
    end
    println(io, "        end")

    println(io, """
    end

    # Compute embeddings with derivatives
    Rnl, dRnl, Ylm, dYlm, rs, rhats = compute_embeddings_ed(Rs, Zs, Z0)

    # Evaluate tensor (using manual inline evaluation, trim-safe)
    # Returns both B and A (A needed for pullback)
    B, A = tensor_evaluate(Rnl, Ylm)

    # Contract with weights to get energy
    # Use pre-allocated ∂B array
    fill!(WORK_∂B, 0.0)
    Ei = 0.0
""")

    # Write weight contraction for virial
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "    $cond iz0 == $iz")
        println(io, "        Ei = dot(B, WB_$iz)")
        println(io, "        for k in 1:N_BASIS; WORK_∂B[k] = WB_$(iz)[k]; end  # ∂Ei/∂B = WB")
    end
    println(io, "    end")

    println(io, """

    # Backward pass through tensor using MANUAL pullback (trim-safe)
    # Use pre-allocated arrays for gradients
    ∂Rnl = view(WORK_∂Rnl, 1:nneigh, :)
    ∂Ylm = view(WORK_∂Ylm, 1:nneigh, :)
    @inbounds for j in 1:nneigh
        @simd for t in 1:N_RNL; WORK_∂Rnl[j, t] = 0.0; end
        @simd for t in 1:N_YLM; WORK_∂Ylm[j, t] = 0.0; end
    end
    tensor_pullback!(∂Rnl, ∂Ylm, WORK_∂B, Rnl, Ylm, A)

    # Assemble forces and virial
    forces = Vector{SVector{3, Float64}}(undef, nneigh)
    @inbounds for j in 1:nneigh
        f = zero(SVector{3, Float64})
        r = rs[j]
        if r > 1e-10
            rhat = rhats[j]
            Rj = Rs[j]
            # Contribution from radial basis
            for t in 1:N_RNL
                df = (∂Rnl[j, t] * dRnl[j, t]) * rhat
                f = f + df
                # Virial: -Rⱼ ⊗ fⱼ (outer product)
                virial = virial - Rj * df'
            end
            # Contribution from solid harmonics: ∂Ei/∂R_lm * dR_lm/dR
            for t in 1:N_YLM
                df = ∂Ylm[j, t] * dYlm[j, t]
                f = f + df
                virial = virial - Rj * df'
            end
        end
        forces[j] = -f
    end

    # Add reference energy
""")

    # Add E0 for virial
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "    $cond iz0 == $iz; Ei += E0_$iz")
    end
    println(io, "    end")

    println(io, """

    return Ei, forces, virial
end
""")
end

function _write_main(io, NZ)
    println(io, """
# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

function (@main)(ARGS)
    println(Core.stdout, "=== ACE Potential Evaluation ===")
    println(Core.stdout, "Number of species: ", NZ)
    println(Core.stdout, "Basis size: ", N_BASIS)
    println(Core.stdout, "Radial basis size: ", N_RNL)
    println(Core.stdout, "Spherical harmonics: L=", MAXL, " (", N_YLM, " functions)")

    # Example evaluation with test data
    Rs = [
        SVector(2.35, 0.0, 0.0),
        SVector(-0.78, 2.22, 0.0),
        SVector(-0.78, -1.11, 1.92),
    ]
    Zs = fill(I2Z[1], length(Rs))  # Same species as center
    Z0 = I2Z[1]  # Center atom species

    println(Core.stdout, "")
    println(Core.stdout, "Test evaluation:")
    println(Core.stdout, "  Center species: Z=", Z0)
    println(Core.stdout, "  Number of neighbors: ", length(Rs))

    # Energy only
    E = site_energy(Rs, Zs, Z0)
    println(Core.stdout, "  Site energy: ", E, " eV")

    # Analytic forces
    println(Core.stdout, "")
    println(Core.stdout, "Analytic forces:")
    E2, F = site_energy_forces(Rs, Zs, Z0)
    for (j, f) in enumerate(F)
        println(Core.stdout, "  F[", j, "] = [", f[1], ", ", f[2], ", ", f[3], "]")
    end

    # Forces + Virial
    println(Core.stdout, "")
    println(Core.stdout, "With virial stress:")
    E3, F3, V = site_energy_forces_virial(Rs, Zs, Z0)
    println(Core.stdout, "  Energy: ", E3, " eV")
    println(Core.stdout, "  Virial tensor:")
    println(Core.stdout, "    [", V[1,1], ", ", V[1,2], ", ", V[1,3], "]")
    println(Core.stdout, "    [", V[2,1], ", ", V[2,2], ", ", V[2,3], "]")
    println(Core.stdout, "    [", V[3,1], ", ", V[3,2], ", ", V[3,3], "]")

    # Verify analytic vs finite difference forces
    println(Core.stdout, "")
    println(Core.stdout, "Force verification (analytic vs finite difference):")
    h = 1e-5
    max_err = 0.0
    for j in 1:length(Rs)
        f_fd = zeros(3)
        for α in 1:3
            Rs_p = copy(Rs)
            Rs_m = copy(Rs)
            e_α = zeros(3); e_α[α] = h
            Rs_p[j] = Rs[j] + SVector{3}(e_α)
            Rs_m[j] = Rs[j] - SVector{3}(e_α)
            Ep = site_energy(Rs_p, Zs, Z0)
            Em = site_energy(Rs_m, Zs, Z0)
            f_fd[α] = -(Ep - Em) / (2h)
        end
        err = sqrt(sum((F[j] - SVector{3}(f_fd)).^2))
        max_err = max(max_err, err)
        println(Core.stdout, "  F[", j, "] err = ", err)
    end
    println(Core.stdout, "  Max force error: ", max_err)

    println(Core.stdout, "")
    println(Core.stdout, "Evaluation successful!")

    return 0
end
""")
end

function _write_c_interface(io, NZ)
    println(io, """
# ============================================================================
# C INTERFACE FOR SHARED LIBRARY
# ============================================================================
#
# Two API levels:
# 1. SITE-LEVEL (for LAMMPS): Works with pre-computed neighbor lists
# 2. SYSTEM-LEVEL (for Python/ASE): Computes neighbor list internally

# ============================================================================
# HELPER FUNCTIONS FOR C INTERFACE
# ============================================================================

@inline function c_read_Rij(ptr::Ptr{Cdouble}, nneigh::Int)::Vector{SVector{3, Float64}}
    Rs = Vector{SVector{3, Float64}}(undef, nneigh)
    @inbounds for j in 1:nneigh
        x = unsafe_load(ptr, 3*(j-1) + 1)
        y = unsafe_load(ptr, 3*(j-1) + 2)
        z = unsafe_load(ptr, 3*(j-1) + 3)
        Rs[j] = SVector(x, y, z)
    end
    return Rs
end

@inline function c_read_species(ptr::Ptr{Cint}, n::Int)::Vector{Int}
    species = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        species[i] = unsafe_load(ptr, i)
    end
    return species
end

@inline function c_write_forces!(ptr::Ptr{Cdouble}, forces::Vector{SVector{3, Float64}})
    @inbounds for j in 1:length(forces)
        unsafe_store!(ptr, forces[j][1], 3*(j-1) + 1)
        unsafe_store!(ptr, forces[j][2], 3*(j-1) + 2)
        unsafe_store!(ptr, forces[j][3], 3*(j-1) + 3)
    end
end

@inline function c_write_virial!(ptr::Ptr{Cdouble}, virial::SMatrix{3,3,Float64,9})
    # Voigt notation: xx, yy, zz, yz, xz, xy (LAMMPS convention)
    unsafe_store!(ptr, virial[1,1], 1)  # xx
    unsafe_store!(ptr, virial[2,2], 2)  # yy
    unsafe_store!(ptr, virial[3,3], 3)  # zz
    unsafe_store!(ptr, virial[2,3], 4)  # yz
    unsafe_store!(ptr, virial[1,3], 5)  # xz
    unsafe_store!(ptr, virial[1,2], 6)  # xy
end

# ============================================================================
# SITE-LEVEL C INTERFACE (for LAMMPS)
# ============================================================================
# These work directly with LAMMPS neighbor lists.
# Forces returned are forces ON the neighbors (not on the center atom).
# LAMMPS handles force accumulation via Newton's 3rd law.

Base.@ccallable function ace_site_energy(
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble}
)::Cdouble
    if nneigh == 0
        # Return E0 for isolated atom
        iz0 = z2i(z0)
""")

    # Generate E0 lookup for each species
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "        $cond iz0 == $iz; return E0_$iz")
    end
    println(io, "        end")
    println(io, "        return 0.0")
    println(io, "    end")

    println(io, """

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))

    return site_energy(Rs, Zs, Int(z0))
end

Base.@ccallable function ace_site_energy_forces(
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble},
    forces::Ptr{Cdouble}
)::Cdouble
    if nneigh == 0
        iz0 = z2i(z0)
""")

    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "        $cond iz0 == $iz; return E0_$iz")
    end
    println(io, "        end")
    println(io, "        return 0.0")
    println(io, "    end")

    println(io, """

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))

    Ei, Fi = site_energy_forces(Rs, Zs, Int(z0))

    # Write forces (these are -dE/dRj, the force ON neighbor j)
    c_write_forces!(forces, Fi)

    return Ei
end

Base.@ccallable function ace_site_energy_forces_virial(
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble},
    forces::Ptr{Cdouble},
    virial::Ptr{Cdouble}
)::Cdouble
    if nneigh == 0
        iz0 = z2i(z0)
        # Zero virial for isolated atom
        for k in 1:6
            unsafe_store!(virial, 0.0, k)
        end
""")

    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "        $cond iz0 == $iz; return E0_$iz")
    end
    println(io, "        end")
    println(io, "        return 0.0")
    println(io, "    end")

    println(io, """

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))

    Ei, Fi, Vi = site_energy_forces_virial(Rs, Zs, Int(z0))

    c_write_forces!(forces, Fi)
    c_write_virial!(virial, Vi)

    return Ei
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

Base.@ccallable function ace_get_cutoff()::Cdouble
    return RCUT_MAX
end

Base.@ccallable function ace_get_n_species()::Cint
    return Cint(NZ)
end

Base.@ccallable function ace_get_species(idx::Cint)::Cint
    if idx < 1 || idx > NZ
        return Cint(-1)
    end
    return Cint(I2Z[idx])
end

Base.@ccallable function ace_get_n_basis()::Cint
    return Cint(N_BASIS)
end

# ============================================================================
# BASIS EVALUATION (for descriptor computation)
# ============================================================================

Base.@ccallable function ace_site_basis(
    z0::Cint,
    nneigh::Cint,
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble},
    basis_out::Ptr{Cdouble}
)::Cint
    if nneigh == 0
        # Return zeros for isolated atom
        for k in 1:N_BASIS
            unsafe_store!(basis_out, 0.0, k)
        end
        return Cint(0)
    end

    Zs = c_read_species(neighbor_z, Int(nneigh))
    Rs = c_read_Rij(neighbor_Rij, Int(nneigh))

    B = site_basis(Rs, Zs, Int(z0))

    # Write basis to output buffer
    for k in 1:N_BASIS
        unsafe_store!(basis_out, B[k], k)
    end

    return Cint(0)  # Success
end

# ============================================================================
# BATCH API
# ============================================================================
# Process multiple atoms at once, reducing Python-Julia FFI call overhead.
# Note: Threads.@threads doesn't work with --trim=safe (closures are trimmed).
# For multi-threaded evaluation, use LAMMPS (OpenMP) or IPICalculator.

Base.@ccallable function ace_batch_energy_forces_virial(
    natoms::Cint,
    z::Ptr{Cint},
    neighbor_counts::Ptr{Cint},
    neighbor_offsets::Ptr{Cint},
    neighbor_z::Ptr{Cint},
    neighbor_Rij::Ptr{Cdouble},
    energies::Ptr{Cdouble},
    forces::Ptr{Cdouble},
    virials::Ptr{Cdouble}
)::Cvoid
    # Process atoms sequentially
    for i in 1:Int(natoms)
        z0 = unsafe_load(z, i)
        nneigh = Int(unsafe_load(neighbor_counts, i))
        offset = Int(unsafe_load(neighbor_offsets, i))  # 0-indexed from C

        if nneigh == 0
            # Isolated atom - return E0
            iz0 = z2i(z0)
""")

    # Generate E0 lookup for each species for batch API
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, "            $cond iz0 == $iz; E0 = E0_$iz")
    end
    println(io, """            else; E0 = 0.0
            end
            unsafe_store!(energies, E0, i)
            # Zero virial
            for k in 1:6
                unsafe_store!(virials, 0.0, (i-1)*6 + k)
            end
        else
            # Read neighbor data for this atom
            Zs = Vector{Int}(undef, nneigh)
            Rs = Vector{SVector{3, Float64}}(undef, nneigh)

            @inbounds for j in 1:nneigh
                idx = offset + j  # 1-indexed from 0-indexed offset
                Zs[j] = unsafe_load(neighbor_z, idx)
                x = unsafe_load(neighbor_Rij, 3*(idx-1) + 1)
                y = unsafe_load(neighbor_Rij, 3*(idx-1) + 2)
                z_coord = unsafe_load(neighbor_Rij, 3*(idx-1) + 3)
                Rs[j] = SVector(x, y, z_coord)
            end

            # Compute energy, forces, virial
            Ei, Fi, Vi = site_energy_forces_virial(Rs, Zs, Int(z0))

            # Write outputs
            unsafe_store!(energies, Ei, i)

            # Forces for this atom's neighbors
            @inbounds for j in 1:nneigh
                idx = offset + j
                unsafe_store!(forces, Fi[j][1], 3*(idx-1) + 1)
                unsafe_store!(forces, Fi[j][2], 3*(idx-1) + 2)
                unsafe_store!(forces, Fi[j][3], 3*(idx-1) + 3)
            end

            # Virial in Voigt notation: xx, yy, zz, yz, xz, xy
            vbase = (i-1)*6
            unsafe_store!(virials, Vi[1,1], vbase + 1)
            unsafe_store!(virials, Vi[2,2], vbase + 2)
            unsafe_store!(virials, Vi[3,3], vbase + 3)
            unsafe_store!(virials, Vi[2,3], vbase + 4)
            unsafe_store!(virials, Vi[1,3], vbase + 5)
            unsafe_store!(virials, Vi[1,2], vbase + 6)
        end
    end
    return nothing
end

""")
end

# Fallback for non-spline basis
function _write_radial_basis(io, rbasis, NZ)
    error("Only SplineRnlrzzBasis is currently supported for export. Use splinify_first=true or call splinify() on your model first.")
end

# Additional envelope types
function _write_envelope(io, env, iz, jz)
    error("Envelope type $(typeof(env)) not yet supported for export")
end
