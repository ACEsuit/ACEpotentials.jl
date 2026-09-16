# Radial basis and spherical harmonics writing functions
# Split from export_ace_model.jl for maintainability

# The ordered/symmetric species-pair index helpers `_ordered_pairs` and `_sym_pair_index`
# that every per-pair table writer below uses live in splinify.jl, which `extract_hermite_
# spline_data` also needs them from and which export_ace_model.jl includes first.  See the
# comment block there for the convention.

"""
    _agnesi_pair_rcut(p, rmax) -> Float64

The distance at which `ET.eval_agnesi(r, p)` first reaches `+1`, i.e. the cutoff that the
stored Agnesi parameter tuple `p` encodes.  The tuple has no `rcut` field, but
`ET.agnesi_params` builds `b0, b1` from `xin = x(rin)` and `xcut = x(rcut)` so that those two
radii map to -1 and +1; inverting the linear part gives `xcut = (1 - b0) / b1`, and `x(r)` is
strictly decreasing, so a bisection on `s` recovers `rcut` exactly.

`rmax` is returned for a degenerate tuple (one whose `xcut` is outside `(0, 1)`), which means
"do not treat this pair as short-ranged" -- the caller only ever uses the result to detect
pairs whose range falls SHORT of the global cutoff.
"""
function _agnesi_pair_rcut(p, rmax::Real)
    xcut = (1 - p.b0) / p.b1
    (xcut <= 0 || xcut >= 1) && return Float64(rmax)
    g = 1 / xcut - 1                     # = a s^pin / (1 + s^(pin-pcut)) at r = rcut
    f(s) = p.a * s^p.pin / (1 + s^(p.pin - p.pcut)) - g    # strictly increasing in s > 0
    lo, hi = 0.0, 1.0
    while f(hi) < 0 && hi < 1e6
        hi *= 2
    end
    f(hi) < 0 && return Float64(rmax)
    for _ = 1:200
        mid = 0.5 * (lo + hi)
        f(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return p.rin + 0.5 * (lo + hi) * (p.req - p.rin)
end

"""
    _check_hermite_uniform_cutoffs(agnesi_params, NZ, rcut)

Refuse to emit a `:hermite_spline` export whose species pairs do not all share the cutoff the
neighbour lists are built at.

WHY THIS IS A HARD ERROR, not a warning.  `EquivariantTensors`' spline evaluator clamps the
transformed coordinate to `[x0, x1]` and then reads knots `il+1, il+2` (`_spl_grid`,
`embed/transsplines.jl:200-207`).  At `y == x1` that is knot `NX + 1`.  Any edge with
`rcut[i,j] <= r <= RCUT_MAX` transforms to exactly `y = 1`, so the SPLINIFIED model throws a
`BoundsError` before an exported library can be compared to it.  The export would therefore
be un-verifiable by construction: no 1e-12 gate could ever be run on it.  Emitting an
unverifiable artefact silently is the same class of defect as silently dropping the pair term.

WHAT THE MESSAGE MAY AND MAY NOT SUGGEST.  Reaching this function implies the model IS
splinified: `export_ace_model` demotes `:hermite_spline` to `:polynomial` for an unsplinified
model, and REFUSES `:polynomial` outright for a splinified one, so
`radial_basis == :hermite_spline` at the call site can only mean "splinified".  Telling such a
caller to "use `radial_basis = :polynomial`" would therefore be useless advice on THIS model
-- that keyword now raises a different error, about the splinification, and still produces no
file.  The two remedies below are the ones that actually work for a caller who can reach this
error.
"""
function _check_hermite_uniform_cutoffs(agnesi_params, NZ::Int, rcut::Real)
    short = Tuple{Int,Int,Int,Float64}[]        # (k, iz, jz, this pair's cutoff)
    for (k, iz, jz) in _ordered_pairs(NZ)
        rc = _agnesi_pair_rcut(agnesi_params[_sym_pair_index(iz, jz, NZ)], rcut)
        rc < rcut * (1 - 1e-9) && push!(short, (k, iz, jz, rc))
    end
    isempty(short) && return nothing
    lines = join(["      pair $k = (iz=$iz, jz=$jz): cutoff $(round(rc, digits = 6)) Å" *
                  " (RCUT_MAX is $rcut Å)" for (k, iz, jz, rc) in short], "
")
    error("""
        export_ace_model: radial_basis=:hermite_spline requires every species pair to share
        one cutoff, and this model's pairs do not:
        $lines
        An edge with rcut[i,j] <= r <= RCUT_MAX transforms to exactly y = 1, where
        EquivariantTensors' spline evaluator (_spl_grid, embed/transsplines.jl:200-207)
        indexes knot NX+1 and throws a BoundsError.  The SPLINIFIED model that such an export
        would have to be verified against therefore cannot be evaluated at all, so the export
        is unverifiable by construction.

        Two things fix this; passing radial_basis=:polynomial is NOT one of them, because
        this model is already splinified and export_ace_model refuses :polynomial for a
        splinified model (splinify() left no polynomial recurrence to emit):

          1. Export the model as it was BEFORE splinify() was applied, with
             radial_basis=:polynomial (the default).  That mode handles per-pair cutoffs
             exactly and is gated at 1e-12 against the fitted model.
          2. Rebuild the model with a single rcut shared by every species pair, then
             re-splinify it, if you specifically need the spline tables.

        See "Radial Basis Export Options" in export/README.md.""")
end

"""
    _emit_pair_dispatch(io, NZ, indent, body)

Emit `if k == 1; <body(1)> elseif k == 2; <body(2)> … end` over the `NZ^2` ORDERED species
pairs, the analogue of `_emit_species_dispatch` for the per-pair tables.  Written as an
`if`-chain on a plain `Int` rather than as a tuple lookup because that is the form
`codegen.jl`'s Hermite dispatchers already use and the form that survives `--trim=safe`.
"""
function _emit_pair_dispatch(io, NZ::Int, indent::String, body::Function)
    for k = 1:NZ^2
        cond = k == 1 ? "if" : "elseif"
        println(io, indent, cond, " k == $k; ", body(k))
    end
    println(io, indent, "end")
end

"""
    _rnl_used(tensor) -> Vector{Int}

The sorted set of `Rnl` (i.e. (n,l)) indices that ANY `A` basis function reads, taken from
`tensor.abasis.spec`, whose entries are `(Rnl index, Ylm index)` pairs.

Every row of `Rnl` outside this set is dead weight in the generated code, and provably so
rather than probably so:

  * the forward pass touches `Rnl[j, ϕ1]` only through `ABASIS_SPEC` (`evaluate_abasis!`),
    so a row outside the set never enters `A`, hence never enters `AA`, `B` or the energy;
  * the backward pass increments `∂Rnl[j, ϕ1]` only through the same `ABASIS_SPEC`
    (`pullback_abasis!`) and `∂Rnl` is zeroed before every site, so `∂Rnl[j, t] == 0.0`
    exactly for every `t` outside the set;
  * the force accumulation then adds `∂Rnl[j, t] * dRnl[j, t] * r̂`, which is `0.0 * finite`
    for those rows -- exactly zero, not approximately.

So zeroing those rows changes NO result by a single ulp, which is what lets Task 5 claim
bit-identical parity with the previous generator rather than merely 1e-13 agreement.

It is a real saving, not a theoretical one: the fitted Cantor model has 74 `Rnl` rows of
which the `A` basis reads 44, and the TiAl order-4 model has 92 of which it reads 49.
"""
function _rnl_used(tensor)
    spec = collect(tensor.abasis.spec)
    isempty(spec) && error("abasis spec is empty; nothing to export")
    return sort(unique(Int(ϕ[1]) for ϕ in spec))
end

"""
    _radial_mixing(W_radial, rnl_used) -> (onehot::Bool, rows::Vector{Vector{Int}},
                                           sel::Vector{Vector{Int}})

Decompose the radial mixing weights `W_radial[n_rnl, n_polys, n_pairs]` into the per-ordered-
pair sparsity structure the generated code is emitted from.

`rows[k]` lists the `Rnl` rows that pair `k` actually needs: those that are BOTH nonzero in
`W_radial[:, :, k]` AND read by the `A` basis (`rnl_used`).  `onehot` says whether every pair's
weight block is a selection matrix -- at most one nonzero per row, and every nonzero exactly
`1.0` -- in which case `sel[k][i]` is the single polynomial index feeding `rows[k][i]` and the
mixing collapses from an `n_rnl x n_polys` matrix-vector product to a gather of `length(rows[k])`
entries.  `sel` is empty when `onehot` is false.

`init_Wradial = :onehot` (the setting both benchmark models were fitted with, and the default
for `ACEpotentials.Models.ace_model`) produces exactly such a selection matrix and NO fit
touches it afterwards -- `Wnlq` is not a fitted parameter of the linear model -- so the one-hot
branch is the one that runs in practice.  The dense branch is still emitted, and still tested
(`export/test/test_generator_parity.jl` builds a `:glorot_normal` model for it), because
`init_Wradial` is a user-facing keyword and a learned radial basis is a supported model.
"""
function _radial_mixing(W_radial, rnl_used::AbstractVector{Int})
    n_rnl, n_polys, n_pairs = size(W_radial)
    @assert maximum(rnl_used) <= n_rnl """
        the A basis reads Rnl row $(maximum(rnl_used)) but the radial weights only have
        $n_rnl rows -- abasis.spec and ps.rembed.post.W disagree about the radial basis size"""
    rows = [Int[] for _ = 1:n_pairs]
    sel  = [Int[] for _ = 1:n_pairs]
    onehot = true
    for k = 1:n_pairs
        Wk = @view W_radial[:, :, k]
        for t in rnl_used
            nz = findall(!=(0.0), Wk[t, :])
            isempty(nz) && continue
            push!(rows[k], t)
            if length(nz) == 1 && Wk[t, nz[1]] == 1.0
                push!(sel[k], nz[1])
            else
                onehot = false
            end
        end
    end
    onehot || (sel = [Int[] for _ = 1:n_pairs])
    return onehot, rows, sel
end

"""
    _polys_used(W_radial, mix_rows, mix_sel, onehot) -> Int

The largest polynomial index the emitted mixing ever reads, i.e. the length the three-term
recurrence has to be evaluated to.

Truncating the recurrence there is EXACT, not approximate, and for a reason worth stating
because it is the whole of Task 6's "width change":

  * the recurrence is strictly forward -- `P[n] = (A[n] y + B[n]) P[n-1] + C[n] P[n-2]` and
    likewise for `dP` -- so `P[1:q]` and `dP[1:q]` are computed from `A/B/C[1:q]` alone and
    do not depend on `N_POLYS` in any way;
  * `P_env` and `dP_env_dr` are formed elementwise from `P`, `dP` and the (scalar) envelope,
    so their first `q` entries are likewise unchanged;
  * the mixing reads `P_env[s]` only for `s` in this set (one-hot) or multiplies by a `W`
    block whose columns beyond it are identically zero (dense).

So every emitted number is bit-identical to the untruncated evaluation.  It is the largest
single per-edge item B1 left behind: the fitted Cantor model evaluates 45 polynomials and
reads 6, TiAl evaluates 33 and reads 11.
"""
function _polys_used(W_radial, mix_rows, mix_sel, onehot::Bool)
    q = 0
    if onehot
        for sel in mix_sel
            isempty(sel) || (q = max(q, maximum(sel)))
        end
    else
        n_polys = size(W_radial, 2)
        for (k, rows) in enumerate(mix_rows)
            isempty(rows) && continue
            for j = 1:n_polys
                any(!=(0.0), @view W_radial[rows, j, k]) && (q = max(q, j))
            end
        end
    end
    # A model whose mixing reads nothing at all would emit a zero-length recurrence; the
    # generated `eval_polys` indexes `POLY_A[1]` unconditionally, so keep at least one term.
    return max(q, 1)
end

function _write_spline_radial_basis_header(io, rcut)
    println(io, """
# ============================================================================
# RADIAL BASIS CONFIGURATION
# ============================================================================

const RCUT_MAX = $(rcut)
""")
end

# Write ETACE radial basis using data-table approach for reduced code generation.
# Uses parameter tables and generic kernel functions instead of per-pair code generation.
#
# Returns `mix_rows::Vector{Vector{Int}}` -- for each ORDERED pair k, the global `Rnl` row
# indices this pair's radial evaluator produces, in the order they occupy the local slots
# `1:length(mix_rows[k])` of the narrow `SVector{M_RNL}` it returns.  Task 6's evaluation
# kernel needs exactly that mapping to build its per-pair A-accumulation blocks, and taking it
# from the same computation that emitted the tables is what keeps the two in step.
function _write_etace_radial_basis(io, etace, ps, agnesi_params, NZ, rcut)
    println(io, """
# ============================================================================
# RADIAL BASIS (ETACE: Data-table approach)
# ============================================================================
""")

    # Write cutoff
    println(io, "const RCUT_MAX = $(rcut)")
    println(io)

    # The ordered species-pair index `pair_idx(iz, jz)` is emitted once by `_write_species`
    # and keys every per-pair table below.

    # Extract polynomial basis info
    rembed_layer = etace.rembed.layer
    poly_basis = rembed_layer.basis.l.layers.layer_1
    n_polys = length(poly_basis)

    # Extract polynomial coefficients
    poly_refstate = poly_basis.refstate
    poly_A = poly_refstate.A
    poly_B = poly_refstate.B
    poly_C = poly_refstate.C

    # Radial weights (needed BEFORE the polynomials are emitted: the recurrence is written at
    # the truncated width `n_polys_used`, which comes out of W's sparsity).
    W_radial = ps.rembed.post.W
    n_rnl = size(W_radial, 1)
    n_pairs = size(W_radial, 3)
    # The per-pair mixing functions are dispatched by `if k == 1 … elseif k == NZ^2`, so the
    # weight tensor must carry exactly one block per ORDERED pair.  (It always has; this
    # turns "always has" into "is checked", since a mismatch would silently drop pairs.)
    @assert n_pairs == NZ^2 """
        radial weights have $n_pairs species-pair blocks, expected NZ^2 = $(NZ^2)
        (one per ORDERED pair, as ET.catcat2idx indexes the SelectLinL weights)"""

    rnl_used = _rnl_used(etace.basis)
    onehot, mix_rows, mix_sel = _radial_mixing(W_radial, rnl_used)
    n_polys_used = _polys_used(W_radial, mix_rows, mix_sel, onehot)
    m_max = maximum(length(r) for r in mix_rows)
    @assert m_max >= 1 """
        no ordered species pair contributes a single (n,l) row -- the exported radial basis
        would be identically zero"""

    println(io, "# Polynomial basis (orthonormalized Chebyshev)")
    println(io, "#")
    println(io, "# TRUNCATED RECURRENCE (Task 6 / B2).  The model carries $(n_polys)")
    println(io, "# polynomials, but the mixing below reads only the first $(n_polys_used):")
    println(io, "# no RBASIS_SEL_k / RBASIS_W_k column beyond that index is ever selected or")
    println(io, "# nonzero.  The recurrence is strictly FORWARD -- P[n] depends only on P[n-1]")
    println(io, "# and P[n-2] -- so stopping at N_POLYS_USED leaves P[1:N_POLYS_USED] and")
    println(io, "# dP[1:N_POLYS_USED] bit-identical to the untruncated evaluation.  The full")
    println(io, "# count is kept as N_POLYS for provenance; nothing evaluates it.")
    println(io, "const N_POLYS = $(n_polys)")
    println(io, "const N_POLYS_USED = $(n_polys_used)  # of N_POLYS = $(n_polys)")
    println(io, "const POLY_A = SVector{$(n_polys_used), Float64}($(repr(collect(poly_A)[1:n_polys_used])))")
    println(io, "const POLY_B = SVector{$(n_polys_used), Float64}($(repr(collect(poly_B)[1:n_polys_used])))")
    println(io, "const POLY_C = SVector{$(n_polys_used), Float64}($(repr(collect(poly_C)[1:n_polys_used])))")
    println(io)

    # Write polynomial evaluation
    println(io, """
# Polynomial evaluation via 3-term recurrence, truncated to N_POLYS_USED (see above)
@inline function eval_polys(y::T) where {T}
    P = MVector{N_POLYS_USED, T}(undef)
    @inbounds begin
        P[1] = T(POLY_A[1])
        if N_POLYS_USED >= 2
            P[2] = POLY_A[2] * y + POLY_B[2]
        end
        for n = 3:N_POLYS_USED
            P[n] = (POLY_A[n] * y + POLY_B[n]) * P[n-1] + POLY_C[n] * P[n-2]
        end
    end
    return P
end

@inline function eval_polys_ed(y::T) where {T}
    P = MVector{N_POLYS_USED, T}(undef)
    dP = MVector{N_POLYS_USED, T}(undef)
    @inbounds begin
        P[1] = T(POLY_A[1])
        dP[1] = zero(T)
        if N_POLYS_USED >= 2
            P[2] = POLY_A[2] * y + POLY_B[2]
            dP[2] = T(POLY_A[2])
        end
        for n = 3:N_POLYS_USED
            P[n] = (POLY_A[n] * y + POLY_B[n]) * P[n-1] + POLY_C[n] * P[n-2]
            dP[n] = POLY_A[n] * P[n-1] + (POLY_A[n] * y + POLY_B[n]) * dP[n-1] + POLY_C[n] * dP[n-2]
        end
    end
    return P, dP
end
""")

    println(io, "const N_RNL = $(n_rnl)")
    println(io)

    # ------------------------------------------------------------------------------------
    # RADIAL MIXING, emitted from W's actual sparsity structure (Task 5 / B1)
    #
    # What this replaces: a per-ordered-pair DENSE `SMatrix{N_RNL, N_POLYS}` and, on every
    # edge, two full matrix-vector products `W * P_env` and `W * dP_env` -- 2 * N_RNL *
    # N_POLYS multiply-adds, 6660 of them per edge on the fitted Cantor model and 6072 on
    # TiAl.  `W` is not dense: it is what `init_Wradial = :onehot` built (a selection matrix)
    # and no fit touches it, so on Cantor 370 of its 83250 entries are nonzero.  Combined
    # with dropping the (n,l) rows the A basis never reads (see `_rnl_used`), 9 of 74 rows
    # survive per Cantor pair and 26 of 92 per TiAl pair.
    #
    # Task 6 / B2 takes the WIDTH with it: `_mix_k` now returns an `SVector{M_RNL}` carrying
    # only the rows this pair populates, in `RBASIS_ROWS_k` order, rather than scattering them
    # back into an `SVector{N_RNL}` of mostly zeros.  M_RNL is the widest such set over all
    # pairs, so every pair's evaluator has one return type and the per-neighbour kernel can
    # cache it in an isbits `NeighCache`.  The full-width `evaluate_Rnl` / `evaluate_Rnl_d`
    # below are kept as scattering wrappers: they are what the tests and the diagnostic
    # scripts compare against the model, and nothing on the hot path calls them.
    # ------------------------------------------------------------------------------------
    println(io, "# The (n,l) rows any A basis function reads (from ABASIS_SPEC). Rows outside")
    println(io, "# this set can never reach the energy or the forces -- see _rnl_used in")
    println(io, "# export/src/write_radial.jl for why zeroing them is exact, not approximate.")
    println(io, "const RNL_USED = $(repr(Tuple(rnl_used)))")
    println(io, "const N_RNL_USED = $(length(rnl_used))  # of N_RNL = $n_rnl")
    println(io, "# The widest per-pair row set: the length of the narrow radial vectors the")
    println(io, "# evaluation kernel passes around and caches.")
    println(io, "const M_RNL = $(m_max)")
    println(io, "# true  -> every pair's W is a selection matrix; the mixing is a gather.")
    println(io, "# false -> W is dense; the mixing is a length(RBASIS_ROWS_k) x N_POLYS_USED GEMV.")
    println(io, "const RBASIS_ONEHOT = $onehot")
    println(io)

    for (k, iz, jz) in _ordered_pairs(NZ)
        rows = mix_rows[k]
        m = length(rows)
        println(io, "# --- pair $k: ($iz, $jz) -- $m of $n_rnl Rnl rows are nonzero and used ---")
        println(io, "const RBASIS_ROWS_$k = SVector{$m, Int}($(repr(rows)))")
        if m == 0
            println(io, "@inline _mix_$k(P_env::SVector{N_POLYS_USED, T}) where {T} = zero(SVector{M_RNL, T})")
            println(io, "@inline _scatter_full_$k(v::SVector{M_RNL, T}) where {T} = zero(SVector{N_RNL, T})")
            println(io)
            continue
        end
        if onehot
            sel = mix_sel[k]
            println(io, "const RBASIS_SEL_$k = SVector{$m, Int}($(repr(sel)))  # polynomial feeding each row")
            entries = ["P_env[$q]" for q in sel]
            println(io, "@inline function _mix_$k(P_env::SVector{N_POLYS_USED, T}) where {T}")
            println(io, "    @inbounds return SVector{M_RNL, T}(")
            println(io, _scatter_expr(collect(1:m), entries, m_max))
            println(io, "    )")
            println(io, "end")
        else
            Wk = W_radial[rows, 1:n_polys_used, k]
            println(io, "const RBASIS_W_$k = SMatrix{$m, $(n_polys_used), Float64, $(m * n_polys_used)}($(repr(vec(Wk))))")
            entries = ["v[$i]" for i = 1:m]
            println(io, "@inline function _mix_$k(P_env::SVector{N_POLYS_USED, T}) where {T}")
            println(io, "    v = RBASIS_W_$k * P_env")
            println(io, "    @inbounds return SVector{M_RNL, T}(")
            println(io, _scatter_expr(collect(1:m), entries, m_max))
            println(io, "    )")
            println(io, "end")
        end
        # Narrow -> full width, for the compatibility wrappers only (never on the hot path).
        println(io, "@inline function _scatter_full_$k(v::SVector{M_RNL, T}) where {T}")
        println(io, "    @inbounds return SVector{N_RNL, T}(")
        println(io, _scatter_expr(rows, ["v[$i]" for i = 1:m], n_rnl))
        println(io, "    )")
        println(io, "end")
        println(io)
    end

    # Write parameter table for all species pairs
    println(io, "# ============================================================================")
    println(io, "# TRANSFORM PARAMETERS (data table approach)")
    println(io, "# ============================================================================")
    println(io)

    println(io, "# Agnesi transform parameters, expanded from the model's per-SYMMETRIC-pair")
    println(io, "# storage to one entry per ORDERED pair, indexed by pair_idx(iz, jz).")
    println(io, "const TRANSFORM_PARAMS = (")

    @assert length(agnesi_params) == (NZ * (NZ + 1)) ÷ 2 """
        many-body transform params: $(length(agnesi_params)) entries, expected \
        $((NZ*(NZ+1))÷2) (one per symmetric pair, as ETModels._convert_agnesi stores them)"""

    for (k, iz, jz) in _ordered_pairs(NZ)
        sym = _sym_pair_index(iz, jz, NZ)
        p = agnesi_params[sym]

        # `pin` and `pcut` are emitted as Int, not Float64.  `s^2` is a multiply; `s^2.0` is a
        # call to libm `pow`, and the transform evaluates three of them per edge (`s^pin`,
        # `s^(pin-pcut)`, `s^(pin-1)`).  This is the same choice `_write_pair_basis` already
        # made for the pair term (see its header comment).  It is not a change of value for
        # the integer exponents these models carry: glibc's `pow` is correctly rounded, and
        # the correctly-rounded `s^2.0` IS `s*s`, so the two agree bit for bit -- which the
        # generator-parity gate measures rather than assumes.  (For an exponent of 3 or more
        # `power_by_squaring` and `pow` may differ by 1 ulp; no model here has one.)
        @assert p.pin isa Integer && p.pcut isa Integer """
            Agnesi transform of pair $k has non-integer pin=$(p.pin) / pcut=$(p.pcut);
            the generated code raises `s` to them as integer powers."""
        println(io, "    (rin=$(Float64(p.rin)), req=$(Float64(p.req)), rcut=$(Float64(rcut)), " *
                    "pin=$(Int(p.pin)), pcut=$(Int(p.pcut)), a=$(Float64(p.a)), " *
                    "b0=$(Float64(p.b0)), b1=$(Float64(p.b1))),  # pair $k: ($iz, $jz) -> sym $sym")
    end
    println(io, ")")
    println(io)

    # Write generic kernel functions
    println(io, """
# ============================================================================
# GENERIC KERNEL FUNCTIONS
# ============================================================================

# Quartic envelope: (1 - y²)²
@inline envelope_quartic(y::T) where {T} = max(zero(T), one(T) - y^2)^2
@inline function envelope_quartic_d(y::T) where {T}
    one_minus_y2 = max(zero(T), one(T) - y^2)
    return one_minus_y2^2, -4 * y * one_minus_y2
end

# Generalized Agnesi transform, mirroring ET.eval_agnesi
# (EquivariantTensors src/transforms/agnesi.jl:45-61):
#     s = (r - rin) / (req - rin);  x = 1/(1 + a s^pin / (1 + s^(pin-pcut)));  y = b1 x + b0
# clamped to [-1, 1].  The parameters are built so that r = rin maps to y = -1 and r = rcut
# to y = +1 (`xin -> -1`, `xcut -> +1` in `agnesi_params`), which is why the shortcuts below
# return -1 at/below rin and +1 at/above rcut.  They are shortcuts for the clamp, NOT an
# independent convention: getting their signs wrong (as this generator did until Task 2)
# is invisible while rin = 0 and no edge reaches rcut, and silently wrong otherwise.
#
# `agnesi_transform` and `agnesi_transform_d` MUST form `s` identically -- both by division,
# as eval_agnesi does.  Forming it as `(r - rin) * (1/(req - rin))` in one of them differs by
# 1 ulp, which the N_POLYS-term recurrence amplifies into a ~4e-14 eV/site disagreement
# between `site_energy` and `site_energy_forces_virial` (they take different routes through
# these two functions).  See export/test/test_pair_export.jl.
@inline function agnesi_transform(r::T, p) where {T}
    if r <= p.rin
        return -one(T)
    end
    if r >= p.rcut
        return one(T)
    end
    s = (r - p.rin) / (p.req - p.rin)
    s_pin = s^p.pin
    s_diff = s^(p.pin - p.pcut)
    denom = one(T) + s_diff
    x = one(T) / (one(T) + p.a * s_pin / denom)
    y = p.b1 * x + p.b0
    return clamp(y, -one(T), one(T))
end

# Agnesi transform with derivative.  `s` is computed exactly as above (division); `ds_dr` is
# formed separately for the chain rule and never used to build `s`.
@inline function agnesi_transform_d(r::T, p) where {T}
    if r <= p.rin
        return -one(T), zero(T)
    end
    if r >= p.rcut
        return one(T), zero(T)
    end

    s = (r - p.rin) / (p.req - p.rin)
    ds_dr = one(T) / (p.req - p.rin)

    s_pin = s^p.pin
    s_diff = s^(p.pin - p.pcut)
    denom = one(T) + s_diff

    x = one(T) / (one(T) + p.a * s_pin / denom)
    y = p.b1 * x + p.b0

    # Derivative via chain rule
    dg_ds = p.a * s^(p.pin - 1) * (p.pin + p.pcut * s_diff) / (denom^2)
    dx_ds = -x^2 * dg_ds
    dy_ds = p.b1 * dx_ds
    dy_dr = dy_ds * ds_dr

    y_clamped = clamp(y, -one(T), one(T))
    if y_clamped != y
        return y_clamped, zero(T)
    end
    return y, dy_dr
end
""")

    # Write generic radial basis functions.  Everything up to `P_env` is pair-generic; only
    # the MIXING is per-pair, and it is reached through the same `if k == …` chain the
    # Hermite dispatchers in codegen.jl use.
    println(io, """
# ============================================================================
# RADIAL BASIS EVALUATION (generic transform + envelope, per-pair mixing)
# ============================================================================

# Narrow radial basis evaluation for any pair: only the rows the pair populates, in
# RBASIS_ROWS_k order, padded to M_RNL.  This is what the per-neighbour kernel calls.
@inline function _evaluate_Rnl_pair_m(r::T, k::Int)::SVector{M_RNL, T} where {T}
    @inbounds p = TRANSFORM_PARAMS[k]
    y = agnesi_transform(r, p)

    env = envelope_quartic(y)
    if env <= zero(T)
        return zero(SVector{M_RNL, T})
    end

    P = eval_polys(y)
    P_env = SVector{N_POLYS_USED, T}(env .* P)""")

    _emit_pair_dispatch(io, NZ, "    ", k -> "return _mix_$k(P_env)")

    # A `k` outside 1:NZ^2 is a BUG in the caller (pair_idx, or the per-neighbour indexing),
    # and it used to be invisible: the old `@inbounds RBASIS_W[k]` was undefined behaviour,
    # and a `return zero(...)` fall-through is worse still -- a silently zero radial basis
    # produces plausible-looking wrong energies instead of a crash.  This is a cold branch
    # (the chain above is exhaustive over the emitted tables), so the compare costs nothing
    # measurable on the hot path.
    println(io, """    error("_evaluate_Rnl_pair_m: species-pair index \$k is outside 1:\$(NZ*NZ)")
end

# Narrow radial basis with derivatives
@inline function _evaluate_Rnl_d_pair_m(r::T, k::Int)::Tuple{SVector{M_RNL, T}, SVector{M_RNL, T}} where {T}
    @inbounds p = TRANSFORM_PARAMS[k]
    y, dy_dr = agnesi_transform_d(r, p)

    env, denv_dy = envelope_quartic_d(y)
    denv_dr = denv_dy * dy_dr

    if env <= zero(T)
        return zero(SVector{M_RNL, T}), zero(SVector{M_RNL, T})
    end

    P, dP = eval_polys_ed(y)
    dP_dr = dP .* dy_dr

    P_env = SVector{N_POLYS_USED, T}(env .* P)
    dP_env_dr = SVector{N_POLYS_USED, T}(denv_dr .* P .+ env .* dP_dr)""")

    _emit_pair_dispatch(io, NZ, "    ", k -> "return _mix_$k(P_env), _mix_$k(dP_env_dr)")

    println(io, """    error("_evaluate_Rnl_d_pair_m: species-pair index \$k is outside 1:\$(NZ*NZ)")
end
""")

    # Full-width scatter dispatch, used ONLY by the compatibility wrappers below.
    println(io, """
# Narrow (M_RNL, pair-local) -> full width (N_RNL, global (n,l) index).
@inline function _scatter_full(v::SVector{M_RNL, T}, k::Int)::SVector{N_RNL, T} where {T}""")
    _emit_pair_dispatch(io, NZ, "    ", k -> "return _scatter_full_$k(v)")
    println(io, """    error("_scatter_full: species-pair index \$k is outside 1:\$(NZ*NZ)")
end
""")

    println(io, """
# Public API, FULL WIDTH.  These are the functions the test suite and the diagnostic scripts
# compare against the model's own `Rnl`, so they keep the N_RNL layout.  The evaluation
# kernel does NOT call them: it uses the narrow `_evaluate_Rnl_*_pair_m` above and the
# per-pair A blocks, which is what removes the N_RNL-wide copies from the hot path.
@inline function evaluate_Rnl(r::T, iz::Int, jz::Int)::SVector{N_RNL, T} where {T}
    k = pair_idx(iz, jz)
    return _scatter_full(_evaluate_Rnl_pair_m(r, k), k)
end

@inline function evaluate_Rnl_d(r::T, iz::Int, jz::Int)::Tuple{SVector{N_RNL, T}, SVector{N_RNL, T}} where {T}
    k = pair_idx(iz, jz)
    v, dv = _evaluate_Rnl_d_pair_m(r, k)
    return _scatter_full(v, k), _scatter_full(dv, k)
end
""")

    return mix_rows
end

# ============================================================================
# PAIR POTENTIAL (ETPairModel)
# ============================================================================
#
# Model (src/et_models/et_pair.jl + src/et_models/convert.jl:`convertpair`):
#
#   rembed  = EdgeEmbed( EnvRBranchL(envelope, EmbedDP(agnesi, polys, SelectLinL)) )
#   readout = SelectLinL(n_pairbasis -> 1, NZ, selector = centre species)
#
#   Rnl_pair[edge, n] = env(r_ij) * Σ_q W[n, q, (iz0,jz)] * P_q(y_ij)
#   𝔹[i, n]           = Σ_{j ∈ N(i)} Rnl_pair[edge, n]          (et_pair.jl:48-57)
#   E_pair(i)         = Σ_n Wread[1, n, iz0] * 𝔹[i, n]          (et_pair.jl:25-33)
#
# so, folding the readout into the polynomial coefficients at export time,
#
#   E_pair(i) = Σ_{j ∈ N(i)} env(r_ij) * dot( PAIR_C[(iz0,jz)], P(y_ij) )
#   PAIR_C[(iz0,jz)][q] = Σ_n Wread[1, n, iz0] * W[n, q, (iz0,jz)]
#
# Index conventions, all verified against the sources rather than guessed:
#  * `ET.catcat2idx` (utils/selector.jl) = (i1-1)*NZ + i2 with i1 the *centre* species
#    (the graph stores z0 = species(i), z1 = species(j); EquivariantTensors
#    ext/NeighbourListsExt.jl:19-21), so the SelectLinL weights W and PAIR_C are indexed by
#    the ORDERED pair (iz0, jz) -- same convention as the many-body RBASIS_W above.
#  * the Agnesi transform parameters are stored per SYMMETRIC pair
#    (`_convert_agnesi` loops `for i = 1:NZ, j = i:NZ` and the selector is
#    `catcat2idx_sym`), i.e. NZ*(NZ+1)/2 entries addressed by `symidx`.  The ordered ->
#    symmetric mapping below goes through the shared `_sym_pair_index` helper (splinify.jl),
#    exactly as TRANSFORM_PARAMS does.
#  * the readout weight Wread is per CENTRE species only (shape (1, n_pairbasis, NZ)).
#
# Numerics:
#  * the polynomials are the raw P4ML `OrthPolyBasis1D3T` -- unlike the many-body radial
#    basis the pair basis has NO quartic envelope wrapped around them; the envelope is the
#    separate `PolyEnvelope1sR` branch.
#  * the envelope is `_eval_env_1sr` (src/et_models/convert.jl:233-237):
#        env(r) = (s^-p - 1) * (1 - s) * (s < 1),  s = r / rcut
#  * the transform is `ET.eval_agnesi` (EquivariantTensors src/transforms/agnesi.jl:53-61).
#    A dedicated `_pair_transform_d` is emitted rather than reusing `agnesi_transform_d`
#    because the latter carries `r <= rin -> +1` / `r >= rcut -> -1` shortcuts that
#    `eval_agnesi` does not have (it only clamps), and because the stored parameter tuple
#    has no `rcut` field of its own.  `pin`/`pcut` are kept as `Int` so that `s^pin` is the
#    same *integer* power `eval_agnesi` evaluates (`s^4` by squaring, not `pow(s, 4.0)`).
#    That is not full bit-exactness: `_pair_transform_d` forms `s` by reciprocal-multiply
#    (it needs `ds/dr` anyway) where `eval_agnesi` divides, which can differ by 1 ulp.
#
# `etace_zlist` and `rcut` come from the ETACE model -- they are what the generated `NZ`,
# `z2i` and `RCUT_MAX` are built from -- and are passed in only so they can be checked
# against the pair model's own species ordering and cutoff.
function _write_pair_basis(io, pair_calc, NZ, etace_zlist, rcut)
    pm, ps = pair_calc.model, pair_calc.ps

    branch = pm.rembed.layer            # EnvRBranchL(envelope, rbasis)
    rb     = branch.rbasis              # EmbedDP(trans, basis, post)
    polys  = rb.basis                   # Polynomials4ML.OrthPolyBasis1D3T
    pA, pB, pC = polys.refstate.A, polys.refstate.B, polys.refstate.C
    nq = length(pA)

    W     = ps.rembed.rbasis.post.W     # (n_pairbasis, n_pairpolys, NZ^2)
    Wr    = ps.readout.W                # (1, n_pairbasis, NZ)
    env   = branch.envelope.refstate    # (rcut, p) of the PolyEnvelope1sR branch
    trans = rb.trans.refstate.params    # SVector{NZ(NZ+1)/2} of Agnesi parameters

    # The generated NZ / z2i / RCUT_MAX are built from the *ETACE* model, while W, Wr and
    # `trans` are indexed by the *pair* model's own species ordering.  Size checks alone pass
    # under any permutation of the species, so compare the orderings themselves: a pair basis
    # whose `_i2z` differs from the many-body one would otherwise export a silently permuted
    # PAIR_C / PAIR_TRANSFORM_PARAMS.
    pair_zs  = [Int(z.atomic_number) for z in rb.trans.refstate.zlist]
    etace_zs = [Int(z.atomic_number) for z in etace_zlist]
    @assert pair_zs == etace_zs """
        pair and many-body species orderings differ -- the exported pair weights would be
        permuted relative to the generated z2i.
          ETACE zlist (defines NZ and z2i) : $etace_zs
          pair  zlist (indexes W and trans): $pair_zs"""

    n_pairbasis = size(W, 1)
    @assert size(W, 2) == nq "pair SelectLinL in_dim $(size(W,2)) != n polys $nq"
    @assert size(W, 3) == NZ^2 "pair SelectLinL has $(size(W,3)) categories, expected NZ^2 = $(NZ^2)"
    @assert size(Wr) == (1, n_pairbasis, NZ) "pair readout W has size $(size(Wr)), expected (1, $n_pairbasis, $NZ)"
    @assert length(trans) == (NZ * (NZ + 1)) ÷ 2 "pair transform params: $(length(trans)) entries, expected $((NZ*(NZ+1))÷2) (per symmetric pair)"
    # The neighbour lists the generated code is driven with are built at RCUT_MAX, so a pair
    # envelope reaching further would be silently truncated.
    @assert env.rcut <= rcut """
        pair envelope cutoff $(env.rcut) Å exceeds the exported RCUT_MAX $(rcut) Å; the pair
        term would be silently truncated by the neighbour list."""

    println(io, """
# ============================================================================
# PAIR POTENTIAL (ETPairModel; readout folded into per-ordered-pair coefficients)
# ============================================================================
""")

    println(io, "# Orthogonal polynomial basis of the pair term (3-term recurrence)")
    println(io, "const N_PAIRPOLYS = $(nq)")
    println(io, "const PAIRPOLY_A = SVector{$(nq), Float64}($(repr(collect(pA))))")
    println(io, "const PAIRPOLY_B = SVector{$(nq), Float64}($(repr(collect(pB))))")
    println(io, "const PAIRPOLY_C = SVector{$(nq), Float64}($(repr(collect(pC))))")
    println(io)

    println(io, "# PolyEnvelope1sR:  env(r) = (s^-p - 1) * (1 - s) * (s < 1),  s = r / rcut")
    println(io, "const PAIR_ENV_RCUT = $(Float64(env.rcut))")
    println(io, "const PAIR_ENV_P = $(Int(env.p))")
    println(io)

    println(io, "# Readout-folded polynomial coefficients, one entry per ORDERED pair,")
    println(io, "# indexed by pair_idx(iz0, jz) with iz0 the CENTRE species.")
    println(io, "const PAIR_C = (")
    for (k, iz0, jz) in _ordered_pairs(NZ)
        c = vec(transpose(Wr[1, :, iz0]) * W[:, :, k])
        @assert length(c) == nq
        println(io, "    SVector{$(nq), Float64}($(repr(collect(c)))),  # pair $k: ($iz0, $jz)")
    end
    println(io, ")")
    println(io)

    println(io, "# Agnesi transform parameters, expanded from the symmetric-pair storage")
    println(io, "# to one entry per ORDERED pair, indexed by pair_idx(iz0, jz) -- the same")
    println(io, "# table layout as PAIR_C above, so no runtime mapping is needed.")
    println(io, "const PAIR_TRANSFORM_PARAMS = (")
    for (k, iz0, jz) in _ordered_pairs(NZ)
        sym_idx = _sym_pair_index(iz0, jz, NZ)
        p = trans[sym_idx]
        println(io, "    (pin=$(Int(p.pin)), pcut=$(Int(p.pcut)), a=$(Float64(p.a)), " *
                    "b0=$(Float64(p.b0)), b1=$(Float64(p.b1)), rin=$(Float64(p.rin)), " *
                    "req=$(Float64(p.req))),  # pair $k: ($iz0, $jz) -> sym $sym_idx")
    end
    println(io, ")")
    println(io)

    println(io, raw"""
# Pair envelope with derivative d/dr
@inline function _pair_env_d(r::Float64)
    s = r / PAIR_ENV_RCUT
    s >= 1.0 && return 0.0, 0.0
    sp = s^(-PAIR_ENV_P)
    e = (sp - 1.0) * (1.0 - s)
    de = (-PAIR_ENV_P * sp / s) * (1.0 - s) - (sp - 1.0)
    return e, de / PAIR_ENV_RCUT
end

# Generalized Agnesi transform of the pair term, with derivative d/dr.
# Mirrors ET.eval_agnesi exactly: no rin/rcut shortcuts, clamp to [-1, 1] only.
@inline function _pair_transform_d(r::Float64, p)
    ds_dr = 1.0 / (p.req - p.rin)
    s = (r - p.rin) * ds_dr
    s_pin = s^p.pin
    s_diff = s^(p.pin - p.pcut)
    denom = 1.0 + s_diff
    x = 1.0 / (1.0 + p.a * s_pin / denom)
    y = p.b1 * x + p.b0
    dg_ds = p.a * s^(p.pin - 1) * (p.pin + p.pcut * s_diff) / (denom * denom)
    dy_dr = p.b1 * (-x * x * dg_ds) * ds_dr
    y_clamped = clamp(y, -1.0, 1.0)
    y_clamped != y && return y_clamped, 0.0
    return y, dy_dr
end

# Pair site-energy contribution of one neighbour, and its derivative w.r.t. r.
#   (iz0, jz) is the ORDERED pair: centre species first.
@inline function pair_energy_d(r::Float64, iz0::Int, jz::Int)
    e, de = _pair_env_d(r)
    e == 0.0 && return 0.0, 0.0
    k = pair_idx(iz0, jz)
    @inbounds p = PAIR_TRANSFORM_PARAMS[k]
    y, dy_dr = _pair_transform_d(r, p)
    @inbounds c = PAIR_C[k]
    @inbounds begin
        P1 = PAIRPOLY_A[1]
        dP1 = 0.0
        P2 = PAIRPOLY_A[2] * y + PAIRPOLY_B[2]
        dP2 = PAIRPOLY_A[2]
        v = c[1] * P1 + c[2] * P2
        dv = c[2] * dP2
        for n = 3:N_PAIRPOLYS
            Pn = (PAIRPOLY_A[n] * y + PAIRPOLY_B[n]) * P2 + PAIRPOLY_C[n] * P1
            dPn = PAIRPOLY_A[n] * P2 + (PAIRPOLY_A[n] * y + PAIRPOLY_B[n]) * dP2 +
                  PAIRPOLY_C[n] * dP1
            v += c[n] * Pn
            dv += c[n] * dPn
            P1, P2, dP1, dP2 = P2, Pn, dP2, dPn
        end
    end
    return e * v, de * v + e * dv * dy_dr
end

@inline pair_energy(r::Float64, iz0::Int, jz::Int) = pair_energy_d(r, iz0, jz)[1]
""")
end

# Stub emitted when the exported model has no ETPairModel term, so that the generated
# evaluation functions are identical in both cases.  Returning literal zeros makes every
# pair contribution an exact no-op (x + 0.0 == x, x + 0.0 * r̂ == x).
function _write_no_pair_basis(io)
    println(io, """
# ============================================================================
# PAIR POTENTIAL: none in this model (many-body + E0 only)
# ============================================================================

@inline pair_energy_d(r::Float64, iz0::Int, jz::Int) = (0.0, 0.0)
@inline pair_energy(r::Float64, iz0::Int, jz::Int) = 0.0
""")
end


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

# Fallback for non-spline basis
function _write_radial_basis(io, rbasis, NZ)
    error("Only SplineRnlrzzBasis is currently supported for export. Use splinify_first=true or call splinify() on your model first.")
end

# Additional envelope types
function _write_envelope(io, env, iz, jz)
    error("Envelope type $(typeof(env)) not yet supported for export")
end
