# Export an ETACE model to trim-compatible Julia code
#
# Usage:
#   include("export_ace_model.jl")
#   export_ace_model(calc, "my_model.jl")
#
# The generated file can be compiled with:
#   juliac --output-lib libace.so --trim=safe model.jl

using ACEpotentials
using ACEpotentials.ETModels: ETACEPotential, ETACE, StackedCalculator, ETOneBody, ETPairModel
using StaticArrays
using SparseArrays
using LinearAlgebra
using Polynomials4ML
const P4ML = Polynomials4ML
using EquivariantTensors
const ET = EquivariantTensors
using AtomsBase: ChemicalSpecies

# Include the pair-index helpers and the code generators
include("build_stamp.jl")
include("pair_index.jl")
include("codegen.jl")
include("symmprod_dag.jl")   # AA product DAG (export time only; see its header)

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

Export a StackedCalculator (e.g., ETOneBody + ETPairModel + ETACE) to a trim-compatible
Julia file.

Automatically extracts E0 values from any ETOneBody calculator, the pair term from any
ETPairModel calculator, and the main ETACE model.

Any other calculator in the stack raises: silently dropping a term produces a library whose
energies and forces are wrong with no indication that anything is missing.
"""
function export_ace_model(calc::StackedCalculator, filename::String; kwargs...)
    # Find ETOneBody, ETPairModel and ETACE calculators in the stack.
    # Anything else is a hard error: an unexported term is an incorrect potential.
    e0_calc = nothing
    pair_calc = nothing
    etace_calc = nothing

    for subcalc in calc.calcs
        m = subcalc.model
        if isa(m, ETOneBody)
            e0_calc = subcalc
        elseif isa(m, ETPairModel)
            pair_calc = subcalc
        elseif isa(m, ETACE)
            etace_calc = subcalc
        else
            error("export_ace_model: cannot export a $(typeof(m)) calculator " *
                  "(only ETOneBody, ETPairModel and ETACE are supported)")
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

    # Call the main export function with E0 values and the pair term
    return export_ace_model(etace_calc, filename;
                            E0_dict=E0_dict, pair_calc=pair_calc, kwargs...)
end


"""
    export_ace_model(calc::ETACEPotential, filename::String; for_library=false, radial_basis=:polynomial)

Export an ETACEPotential to a trim-compatible Julia file.

Arguments:
- `calc`: The fitted ETACEPotential to export
- `filename`: Output filename
- `for_library=false`: If true, generate a shared library with C interface instead of executable
- `radial_basis=:polynomial`: VESTIGIAL -- `:polynomial` is the only mode. See below.
- `aa_products=:flat`: how the AA (symmetric product) step is evaluated -- see below.

# AA product modes (`aa_products`), and why `:flat` is the default

| mode | tensor step | Cantor (order 3, 201 neigh) | TiAl (order 4, 112 neigh) |
|---|---|---|---|
| `:flat` (**default**) | flat AA products, sparse `A2Bmap`, `dot(B, WB_iz)`, flat pullback | **58.1 µs/site** | **94.0 µs/site** |
| `:dag` | binary product DAG + per-species `CTILDE` (the readout folded at export time) | 76.4 µs/site (**1.32x SLOWER**) | not gateable at 1e-12; would be ~1.8x FASTER |

`:dag` replaces the flat AA products with `EquivariantTensors`' `SparseSymmProdDAG`
construction (ported into `export/src/symmprod_dag.jl` -- read its header for why it is ported
rather than imported) and folds `A2Bmapᵀ · WB_iz`, a per-species CONSTANT the `:flat` kernel
recomputes at every site, into `CTILDE_iz`.  The site energy becomes `dot(CTILDE_iz, AAd)` and
the backward pass is two FMAs per node seeded by the same vector, so `B`, the A2B maps and
`WB` leave the energy/force path entirely.

**It makes the tensor step much faster and the Cantor model slower, and both were measured**
(Task 7; `export/bench/README.md`). Per-phase, pinned, pure Julia, one site of the benchmark
models:

| phase | Cantor `:flat` | Cantor `:dag` | TiAl `:flat` | TiAl `:dag` |
|---|---|---|---|---|
| embed (pass 1) | 20.82 µs | 31.39 | 10.64 | 10.27 |
| **tensor step** | 13.49 | **10.98** | 61.92 | **25.18** |
| forces (pass 2) | 28.50 | 42.67 | 16.70 | 15.54 |
| whole site | 49.60 | 65.18 | 83.57 | **46.00** |

Regenerated with `export/bench/profile_tensor_step.jl` (log
`bench_parity/profile_tensor_step.log`).

Passes 1 and 2 are BYTE-IDENTICAL code in the two exports. On TiAl they do not move at all;
on Cantor they get 27-29 % slower, because a 201-neighbour site runs them either side of the
DAG, whose 48 kB of gathered/scattered `AAd`/`∂AAd`/`CTILDE` traffic evicts the per-edge
tables they depend on. TiAl has 112 neighbours and a tensor step that is 73 % of the site, so
the DAG's 2.3x there wins outright.

`:dag` additionally cannot be gated at the plan's 1e-12 absolute force tolerance on the TiAl
order-4 model: re-associating the products moves the exported forces by ~1.7e-12 eV/Å against
the `ETACEPotential` reference. That is NOT a defect of the DAG -- measured against a BigFloat
evaluation of the same expressions, both routes' `∂A` sits at `Σ|terms|·eps ≈ 1.2e-12` for
that model, and `:flat` meets the gate only because it shares the reference's association.
On Cantor, whose `∂A` conditioning is 5-10x milder, `:dag` is measurably the MORE accurate of
the two (0.5-2.1x the cancellation floor against `:flat`'s 2.3-3.4x).

So: `:dag` is the right structure for a high-order, many-function, modest-neighbour-count
model and the wrong one here, and it is available but off.

# Radial basis modes (`radial_basis`) -- VESTIGIAL: `:polynomial` is the only mode

`radial_basis` is kept only because every existing call site passes it, and it accepts
`:polynomial` and nothing else.  Omit it.

| mode | reproduces | accuracy | status |
|---|---|---|---|
| `:polynomial` (**default**) | the fitted model | exact: 1e-12 in energies, forces and virial | the only mode |
| `:hermite_spline` | -- | -- | **REMOVED**; raises |

`:polynomial` re-evaluates the orthogonal polynomial recurrence at runtime and reproduces the
model it was exported from to double-precision roundoff.  It is the mode every verification
step of this project gates at 1e-12, and it handles a dense (i.e. genuinely learned)
radial-mixing tensor and per-pair cutoffs exactly.

WHY `:hermite_spline` WENT, and what it means for a splinified model.  It emitted the knot
tables of a model already splinified with `ETModels.splinify`.  By the time the per-neighbour
kernel landed it had lost every advantage it was kept for: it was **slower** (62.5 vs 58.1
µs/site on Cantor, 152.5 vs 94.0 on TiAl, one pinned core), it was **approximate** (2.7e-4
eV/Å on Cantor and 1.6e-2 eV/Å on TiAl at `Nspl = 50`, against an exact mode gated at 1e-12),
and it was **refused outright for per-pair cutoffs**, because the splinified reference model
itself throws a `BoundsError` in upstream `EquivariantTensors._spl_grid` at `y = 1`.  Its
stated justification -- "learned radials with a small `N_POLYS`" -- was empty: `:polynomial`
emits an arbitrary dense `W` exactly and emits the recurrence only at the width a model
actually reads (45 -> 6 terms on Cantor, 33 -> 11 on TiAl).  The evidence is in
`export/bench/FINDINGS_parity.md` §7; the maintainer answered "remove".

**A SPLINIFIED MODEL CAN NO LONGER BE EXPORTED AT ALL, and that is deliberate.**
`splinify()` replaces the polynomial recurrence with knot tables, so `:polynomial` has nothing
to emit and refuses -- it always did.  `:hermite_spline` was the only path such a model had,
so removing the mode retires the fit-on-splines deployment route with it.  The workflow that
works is: keep the UNSPLINIFIED model, fit it, and export that with `:polynomial` (the
default).  `ACEpotentials.ETModels.splinify` itself is untouched and still useful for
in-Julia evaluation; it is only its export path that is gone.

# Example: export (no pre-processing needed, and none wanted)
```julia
calc = ETACEPotential(etace, ps_fitted, st_fitted, 5.5)
export_ace_model(calc, "my_model.jl"; for_library=true)
# Compile with: juliac --output-lib libace.so --trim=safe my_model.jl
```
"""
function export_ace_model(calc::ETACEPotential, filename::String;
                          for_library::Bool=false,
                          radial_basis::Symbol=:polynomial,
                          aa_products::Symbol=:flat,
                          E0_dict::Union{Dict{Int,Float64},Nothing}=nothing,
                          pair_calc=nothing)
    aa_products in (:flat, :dag) ||
        error("export_ace_model: aa_products must be :flat or :dag, got $aa_products")

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

    # Validate `radial_basis`, and refuse a splinified model.
    #
    # `radial_basis` is VESTIGIAL.  `:hermite_spline` was removed (evidence and the
    # maintainer's answer: export/bench/FINDINGS_parity.md §7), and with it the only export
    # path a splinified model had.  The keyword survives because every existing call site
    # passes it; it accepts `:polynomial` and nothing else.
    #
    # Both refusals below name the workflow that DOES work, because a removal message whose
    # only content is "that is gone" costs the reader the same search twice.
    if radial_basis === :hermite_spline
        error("""
        radial_basis=:hermite_spline has been REMOVED.  There is no spline export.

        Why: by the time the per-neighbour kernel landed, the Hermite mode was SLOWER than
        the exact :polynomial mode on both reference models (62.5 vs 58.1 µs/site on Cantor,
        152.5 vs 94.0 on TiAl), APPROXIMATE by construction (2.7e-4 eV/Å on Cantor and
        1.6e-2 eV/Å on TiAl at Nspl=50, against a mode gated at 1e-12), and REFUSED outright
        for per-pair cutoffs because the splinified reference model itself throws a
        BoundsError in upstream EquivariantTensors._spl_grid.  Its stated justification
        ("learned radials with a small N_POLYS") was empty: :polynomial emits an arbitrary
        dense -- i.e. genuinely learned -- radial-mixing tensor exactly.  The measurements
        are in export/bench/FINDINGS_parity.md §7.

        What to do instead: export with radial_basis=:polynomial (the default, so just drop
        the keyword) from the model as it was BEFORE splinify() was applied.  That export is
        exact and is gated at 1e-12 against the fitted model throughout this project's test
        suite.  It is faster than the spline mode was.

        Note that this is NOT a rename: the fit-on-splines DEPLOYMENT route is retired, not
        relocated.  See the refusal for an already-splinified model below.""")
    elseif radial_basis !== :polynomial
        error("""
        export_ace_model: unknown radial_basis=$(repr(radial_basis)).  The only value is
        :polynomial, which is the default -- the keyword is vestigial and can be omitted.
        (:hermite_spline was removed; see export/bench/FINDINGS_parity.md §7.)""")
    end

    if is_splinified
        error("""
        This model has already been splinified, and a splinified model can no longer be
        exported at all.

        splinify() replaces the radial polynomial recurrence with cubic-spline knot tables,
        so :polynomial -- the only mode, and the default -- has no polynomial basis left to
        emit.  It always refused such a model.  What has changed is that :hermite_spline,
        the one path a splinified model had, was REMOVED: it was slower AND approximate
        against the exact mode on both reference models, and unusable with per-pair cutoffs.
        The evidence is in export/bench/FINDINGS_parity.md §7.

        THE FIT-ON-SPLINES DEPLOYMENT ROUTE IS THEREFORE RETIRED.  There is no supported way
        to deploy a model through splinify(); do not look for a keyword that re-enables one.

        What to do:

          1. Export the model as it was BEFORE splinify() was applied, with the default
             (radial_basis=:polynomial, or no keyword at all).  That export is exact, gated
             at 1e-12 against the fitted model, handles a dense/learned radial-mixing tensor
             and per-pair cutoffs, and is FASTER than the spline export was.  Keep the
             unsplinified model and its fitted parameters; that is what you deploy.

          2. If the parameters you hold were fitted AFTER splinify() -- so there is no
             unsplinified model carrying them -- refit the unsplinified model.  The spline
             representation those parameters belong to has no exporter any more.

        ACEpotentials.ETModels.splinify itself is untouched and still usable for in-Julia
        evaluation.  It is only the EXPORT path for its output that is gone.""")
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

    # Radial basis spec and weights.
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

    # Readout weights: ps.readout.W[1, n_basis, n_species]
    W_readout = ps.readout.W
    n_basis = size(W_readout, 2)

    # Generate the export file.
    #
    # Written to a buffer first so that a BUILD STAMP can be appended: the id below is the
    # sha256 prefix of everything above the marker line, and the generated library exports it
    # as `ace_build_id()`.  Without it, a `libace*.so` sitting next to a newer `.jl` is
    # indistinguishable from one compiled from it -- and that is not hypothetical, it is the
    # state export/test/build was found in (the model file regenerated by every test run, the
    # library compiled once and left alone), which means every gate that goes through the
    # compiled library was silently testing a different model from the one the Julia-side
    # gates measured.  `export/test/runtests.jl:library_build_id` recomputes the id from the
    # `.jl` and refuses to run the library groups against a mismatch.
    # The AA product DAG and the folded readout.  Built once here and handed to both writers
    # so the constants and the kernel cannot be built from two different DAGs.  Built only
    # when it is going to be emitted: on the TiAl order-4 model it is a few hundred ms of
    # partition search that a :flat export has no use for.
    dag = (aa_products == :dag) ? SymmProdDAG(aa_flat_spec(tensor.aabasis)) : nothing

    let io = IOBuffer()
        _write_header(io, for_library)
        _write_species(io, _i2z)
        _write_tensor(io, tensor, dag, W_readout, NZ)

        # Write the radial basis.
        #
        # The writer returns the per-ORDERED-pair radial row sets: `pair_rows[k][i]` is the
        # global (n,l) index carried in local slot `i` of the narrow `SVector{M_RNL}` that
        # pair k's evaluator returns.  `_write_evaluation_functions` builds its per-pair A
        # blocks from exactly this, so the kernel and the tables cannot drift apart.
        pair_rows = _write_etace_radial_basis(io, etace, ps, agnesi_params, NZ, rcut)

        # Pair potential term (from the ETPairModel of a StackedCalculator, if present).
        # Always emits `pair_energy` / `pair_energy_d` so the evaluation functions below
        # are identical with and without a pair term.
        if pair_calc !== nothing
            _write_pair_basis(io, pair_calc, NZ, zlist, rcut)
        else
            _write_no_pair_basis(io)
        end

        _write_spherical_harmonics(io, maxl)
        _write_etace_weights(io, W_readout, NZ, E0_dict, _i2z)
        _write_evaluation_functions(io, tensor, dag, NZ, pair_calc !== nothing, pair_rows;
                                    aa_products = aa_products)
        if for_library
            _write_c_interface(io, NZ)
        else
            _write_main(io, NZ)
        end

        body = String(take!(io))
        _write_build_stamp(io, body, for_library)
        write(filename, body, String(take!(io)))
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

# Ordered species-pair index (CENTRE species first), the single convention used by EVERY
# per-pair table in this file: TRANSFORM_PARAMS, RBASIS_W, PAIR_C, PAIR_TRANSFORM_PARAMS and
# the Hermite knot tables PAIR_k_F / PAIR_k_G.  It matches ET.catcat2idx, which is what
# indexes the SelectLinL weights of the model this file was generated from.
#
# Parameters the model stores per SYMMETRIC pair (the Agnesi transforms, ET.catcat2idx_sym)
# are expanded into this ordered layout at EXPORT time, so there is no second convention and
# no runtime mapping.  For NZ=2: (1,1)->1, (1,2)->2, (2,1)->3, (2,2)->4.
@inline pair_idx(iz::Int, jz::Int) = (iz - 1) * NZ + jz
""")
end

# NOTE, on the line just above that says "the Hermite knot tables PAIR_k_F / PAIR_k_G":
# that sentence is EMITTED INTO EVERY GENERATED MODEL FILE, and those tables no longer exist
# -- `:hermite_spline` was removed.  It is left wrong ON PURPOSE.  `EXPORT_BUILD_ID` is the
# FNV-1a hash of the emitted body (build_stamp.jl), so editing one character of emitted text
# changes the id of every model file, which in turn invalidates every committed gate manifest
# and every standing timing row that was taken against a library compiled from the old id --
# and the acceptance condition for this removal is that the shipped `:polynomial` source stays
# BYTE-IDENTICAL.  Correct it in the next change that legitimately moves the build id (the
# next generator change that alters emitted code), not in a removal whose whole claim is that
# it moved nothing.

"""
    _write_tensor(io, tensor, dag, W_readout, NZ)

Emit the tensor-step constants.  TWO REPRESENTATIONS OF THE SAME BASIS are written, and they
are used by different entry points:

  * the **DAG** (`N_DAG`, `DAG_NUM1`, `DAG_FIRST`, `DAG_NODES`, `CTILDE_iz`) drives the
    ENERGY/FORCE path.  `E = dot(CTILDE_iz, AAd)` with the readout already folded in, so `B`,
    the A2B map and `WB` are not on that path at all (Task 7 / B3).
  * the **flat** `AABASIS_SPECS_*` / `A2BMAP_*` / `WB_*` drive `site_basis` only, which is
    the `ace_site_basis` C entry point's contract: it returns the N_BASIS-vector `B`, which
    the DAG representation does not compute.  That entry point is exported and documented, so
    the constants behind it are kept rather than silently dropped.  They cost image size, not
    time: nothing on the hot path reads them.

`CTILDE_iz = projectionᵀ · (A2Bmapᵀ · WB_iz)`.  Both factors are per-species constants that
the pre-B3 kernel recomputed at EVERY SITE (`A2Bmapᵀ · WB` was a full pass over the sparse map
to seed `∂AA`, and `A2Bmap · AA` a second one to build `B`).
"""
function _write_tensor(io, tensor, dag, W_readout, NZ)
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

    dag === nothing || _write_dag(io, tensor, dag, W_readout, NZ)
end

function _write_dag(io, tensor, dag, W_readout, NZ)
    nnodes = length(dag.nodes)
    has0 = dag.has0 ? 1 : 0
    first_interior = has0 + dag.num1 + 1
    ninterior = nnodes - first_interior + 1
    @assert ninterior >= 0
    nA = length(tensor.abasis)
    @assert dag.num1 <= nA "DAG leaf count $(dag.num1) exceeds the A basis size $nA"

    # Depth, for the provenance comment (and because a DAG whose depth exploded would be a
    # sign the partition heuristic had gone wrong).
    depth = zeros(Int, nnodes)
    for i in first_interior:nnodes
        n1, n2 = dag.nodes[i]
        depth[i] = 1 + max(depth[n1], depth[n2])
    end

    println(io, """
# ============================================================================
# AA PRODUCT DAG  (Task 7 / B3)
# ============================================================================
#
# Binary DAG over the A basis replacing the flat AA products.  Layout (upstream
# SparseSymmProdDAG's, see export/src/symmprod_dag.jl for why that type is ported rather
# than imported):
#
#   AAd[1]                        = 1.0                       iff DAG_HAS0
#   AAd[$(has0)+i] = A[i]                 for i = 1:DAG_NUM1  (LEAVES -- no node emitted)
#   AAd[i]                        = AAd[n1] * AAd[n2]         for i = DAG_FIRST:N_DAG
#
# `DAG_NODES[j]` is the (n1, n2) of node `DAG_FIRST - 1 + j`; both are < that index, so the
# forward loop is one pass up and the pullback one pass down.  A Vector of Int32 pairs, not a
# Tuple of tuples: the pullback indexes it in REVERSE, and a runtime index into a
# $(ninterior)-element tuple is not something to hand a compiler.
#
# This model: $(nnodes) nodes = $(has0) constant + $(dag.num1) leaves + $(ninterior) interior,
# against $(length(tensor.aabasis)) flat AA functions and $(size(W_readout, 2)) B functions.
# Max depth $(maximum(depth; init = 0)).""")

    println(io, "const N_DAG = $nnodes")
    println(io, "const DAG_NUM1 = $(dag.num1)")
    println(io, "const DAG_HAS0 = $(dag.has0)")
    println(io, "const DAG_FIRST = $first_interior")
    nodes_str = join(("($(Int32(dag.nodes[i][1])),$(Int32(dag.nodes[i][2])))"
                      for i in first_interior:nnodes), ", ")
    println(io, "const DAG_NODES = NTuple{2, Int32}[$nodes_str]")
    println(io)

    println(io, """
# Folded readout: CTILDE_iz[n] is the coefficient of DAG node n in the site energy, i.e.
# `projection' * (A2Bmap' * WB_iz)`.  `E = dot(CTILDE_iz, AAd)`, and the SAME vector seeds the
# pullback (`∂AAd .= CTILDE_iz`), which is what makes the backward pass two FMAs per node with
# no ∂B, no ∂AA scatter and no second traversal of the A2B map.""")
    A2B = tensor.A2Bmaps[1]
    for iz in 1:NZ
        ct = dag_ctilde(dag, A2B' * W_readout[1, :, iz])
        @assert length(ct) == nnodes
        println(io, "const CTILDE_$(iz) = $(repr(ct))")
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



function _write_build_stamp(io, body::AbstractString, for_library::Bool)
    id = _fnv1a64(codeunits(body))
    println(io, BUILD_STAMP_MARKER)
    println(io, """
# A 64-bit FNV-1a hash of the body above.  `export_build_id(<this file>)` recomputes it; a
# compiled library returns it from `ace_build_id()`.  A `.so` whose id differs from the `.jl`
# beside it was compiled from a DIFFERENT model -- do not gate one against the other.
const EXPORT_BUILD_ID = $(repr(id))""")
    if for_library
        println(io, """
# Deliberately a UInt64 and not a string: an integer return needs no Cstring conversion and
# no string constant to survive --trim=safe.
Base.@ccallable function ace_build_id()::Culonglong
    return Culonglong(EXPORT_BUILD_ID)
end""")
    end
    return nothing
end
