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

# Include code generation modules (split for maintainability)
include("write_radial.jl")
include("write_evaluation.jl")
include("write_c_interface.jl")

# Helper to emit species dispatch blocks (reduces code duplication)
# Emits: if iz0 == 1; <body(1)> elseif iz0 == 2; <body(2)> ... end
function _emit_species_dispatch(io, NZ::Int, indent::String, body::Function)
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, indent, cond, " iz0 == $iz; ", body(iz))
    end
    println(io, indent, "end")
end

# Multi-line version: body returns multiple lines as a vector of strings
function _emit_species_dispatch_multi(io, NZ::Int, indent::String, body::Function)
    for iz in 1:NZ
        cond = iz == 1 ? "if" : "elseif"
        println(io, indent, cond, " iz0 == $iz")
        for line in body(iz)
            println(io, indent, "    ", line)
        end
    end
    println(io, indent, "end")
end


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

