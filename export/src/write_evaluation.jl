# Evaluation functions writing
# Split from export_ace_model.jl for maintainability
#
# ============================================================================================
# THE KERNEL THIS EMITS (Task 6 / B2), and why it is shaped this way
# ============================================================================================
#
# Before B2 a site was evaluated in five full-width sweeps over the neighbour list:
#
#   1. `compute_embeddings_ed` copied `evaluate_Rnl_d`'s `SVector{N_RNL}` into a
#      `MAX_NEIGHBORS x N_RNL` global work array, one `@simd for t in 1:N_RNL` per edge --
#      74 stores per edge on the fitted Cantor model, of which 44 rows could ever be read and
#      only 9 were nonzero for that edge's species pair;
#   2. `evaluate_abasis!` looped over EVERY A function and, inside, over EVERY neighbour;
#   3. the A2B contraction and the readout;
#   4. `pullback_abasis!` did the same double loop again, writing `∂Rnl` and `∂Ylm` at full
#      width (which had to be zeroed at full width first);
#   5. the force assembly looped `for t in 1:N_RNL` PLUS `for t in 1:N_YLM` per edge, with a
#      rank-1 `virial -= Rⱼ * df'` INSIDE those loops -- 78 outer products per edge.
#
# Every one of those is O(N_RNL) or O(N_A x nneigh) per edge, independent of how few rows the
# edge's species pair actually populates.  B1 pruned the tables; the pruning bought nothing
# because steps 1, 4 and 5 immediately re-spent it at full width.
#
# B2 evaluates per NEIGHBOUR instead, in two passes over the neighbour list:
#
#   pass 1  `_embed_val!`  -- for each edge: the narrow radial vector `SVector{M_RNL}` (only
#                             the rows that pair populates), the solid harmonics, and an
#                             accumulation into A restricted to that pair's A block.
#   (tensor) `_energy_and_∂A!` -- AA, B, the readout, and the pullback down to ∂A.  Unchanged
#                             in substance from before; Task 7 replaces it with a DAG.
#   pass 2  `_forces_from_∂A!` -- for each edge: the radial vector WITH derivatives and the
#                             harmonics WITH gradients, then one `(g, v)` from that pair's A
#                             block, ONE `f`, ONE rank-1 virial update.
#
# PASS 2 RECOMPUTES THE EMBEDDINGS RATHER THAN CACHING THEM, and that is a deliberate choice
# made twice over.  Caching them (an isbits per-neighbour struct in the workspace) was the
# first design and it is ~95 kB of write-then-read traffic per Cantor site; recomputing costs
# one extra Agnesi transform, one 6-term recurrence and one solid-harmonic evaluation per
# edge, all of which B1 and the width change made cheap.  Decisively, it also removes the only
# VARIABLE-LENGTH buffer from the workspace, which is what lets the workspace itself live in
# the compiled image (see the WORKSPACE POOL note in write_c_interface.jl): a `resize!`d
# `Vector` cannot.
#
# The per-pair A block is `ABLOCK_k`: the A indices `a` whose `Rnl` row belongs to pair `k`'s
# row set, together with the LOCAL slot of that row in the narrow radial vector and the Ylm
# index.  Restricting the accumulation to it is exact and not merely accurate: a row outside
# the pair's set is identically zero in `_evaluate_Rnl_pair_m`'s output (that is what
# `_radial_mixing` / `hermite_pair_rows` select on), so the terms dropped are `0.0 * Ylm`.
#
# WORKSPACE.  All scratch lives in a caller-supplied `Workspace`, so the library is re-entrant:
# the LAMMPS plugin holds one per OpenMP thread and the ASE calculator one per instance.  Every
# buffer in it is sized from the MODEL (N_A, N_AA, N_BASIS) and never from the neighbour count,
# so nothing is ever reallocated and there is no maximum neighbour count -- the 256-neighbour
# cap this export used to carry is gone.  In a compiled library the workspaces are a fixed
# pool built into the image and handed out by index; see write_c_interface.jl for why that is
# the only construction that survives `juliac --trim`.
#
# ============================================================================================
#
# The emitted evaluation functions always call `pair_energy` / `pair_energy_d`; when the
# exported model has no ETPairModel those are the zero stubs from `_write_no_pair_basis`,
# which makes every pair contribution an exact no-op.
#
# `has_pair` therefore has NO semantic effect: it gates nothing, it selects only the wording
# of one comment line in the generated file.  The pair call sites are emitted either way.
# Do not add logic behind it without also removing that guarantee from
# `_write_no_pair_basis`'s contract.
#
# `pair_rows[k]` is the ORDERED pair `k`'s radial row set, in the order the narrow radial
# vector carries it.  It comes from whichever radial writer ran (`_write_etace_radial_basis`
# returns it; `hermite_pair_rows` computes it for the spline mode) -- never recomputed here,
# because a second reading of the same data that drifts out of step would produce a kernel
# that indexes the wrong radial slot and is wrong by a smooth, plausible-looking amount.

"""
    _ablocks(tensor, pair_rows) -> Vector{Vector{Tuple{Int,Int,Int}}}

For each ORDERED species pair `k`, the list of `(a, s, y)` triples driving both the forward
A accumulation and the force assembly:

  * `a` -- index into `A` (and into `∂A`),
  * `s` -- LOCAL slot of that A function's `Rnl` row in pair `k`'s narrow radial vector,
  * `y` -- index into the solid harmonics.

i.e. `A[a] += R[s] * Y[y]` for every edge of species pair `k`, which is exactly the subset of
`ABASIS_SPEC` whose radial row `k` populates.  Every other entry of `ABASIS_SPEC` contributes
`0.0 * Y[y]` for such an edge, so this restriction changes no value.
"""
function _ablocks(tensor, pair_rows)
    spec = collect(tensor.abasis.spec)
    blocks = Vector{Vector{Tuple{Int,Int,Int}}}(undef, length(pair_rows))
    for k in eachindex(pair_rows)
        rows = pair_rows[k]
        slot = Dict{Int,Int}(t => i for (i, t) in enumerate(rows))
        blk = Tuple{Int,Int,Int}[]
        for (a, ϕ) in enumerate(spec)
            t = Int(ϕ[1])
            haskey(slot, t) && push!(blk, (a, slot[t], Int(ϕ[2])))
        end
        blocks[k] = blk
    end
    return blocks
end

"""
    _write_evaluation_functions(io, tensor, dag, NZ, has_pair, pair_rows; aa_products)

`aa_products` selects the TENSOR STEP, and `:flat` is the default.  See the `aa_products`
docstring on `export_ace_model` for the measurements behind that default; the short version is
that the DAG makes the tensor step 1.6-2.3x faster on both benchmark models and still makes
the Cantor model 1.32x SLOWER end to end, because on a 201-neighbour site the DAG's
gather/scatter working set evicts the per-edge tables the two neighbour passes depend on.
"""
function _write_evaluation_functions(io, tensor, dag, NZ, has_pair, pair_rows;
                                     aa_products::Symbol = :flat)
    aa_products in (:flat, :dag) ||
        error("_write_evaluation_functions: aa_products must be :flat or :dag, got $aa_products")
    use_dag = (aa_products == :dag)
    use_dag == (dag !== nothing) ||
        error("_write_evaluation_functions: aa_products=$aa_products with dag=$(dag === nothing ? "nothing" : "a DAG"); " *
              "the caller must build a DAG exactly when it asks for one")
    nA = length(tensor.abasis)
    nAA = length(tensor.aabasis)
    has0 = use_dag && dag.has0 ? 1 : 0
    @assert length(pair_rows) == NZ^2 """
        the radial writer returned $(length(pair_rows)) per-pair row sets, expected
        NZ^2 = $(NZ^2) (one per ORDERED pair)"""

    blocks = _ablocks(tensor, pair_rows)

    println(io, "# Pair potential term in the site energy: " *
                (has_pair ? "ACTIVE (ETPairModel exported above)" :
                            "ABSENT (pair_energy* are zero stubs)"))

    println(io, """
# ============================================================================
# WORKSPACE (no neighbour cap, nothing reallocated -- the library is re-entrant)
# ============================================================================
#
# One Workspace per concurrent caller.  The LAMMPS plugin allocates one per OpenMP thread via
# ace_workspace_new(); the Python calculator holds one per instance.  Two threads with two
# workspaces produce bitwise the serial answer.
#
# EVERY FIELD IS SIZED FROM THE MODEL, never from the neighbour count.  That is what removes
# the 256-neighbour cap this export used to carry -- a site of any size uses the same buffers
# -- and it is also what lets the compiled library keep its workspaces in the image itself
# rather than on the runtime heap (see write_c_interface.jl).

const N_A = $nA
const N_AA = $nAA
""")
    if use_dag
        println(io, """
# `AAd` / `∂AAd` (Task 7 / B3, aa_products = :dag) are sized from N_DAG, which is a MODEL
# constant emitted above -- exactly the same class as N_AA and N_BASIS, evaluated when the
# image is built.  The rule the workspace has to obey is that no field is sized at RUNTIME
# (that is what would need a `resize!` and is what the Task 6 report's section 3 measured as
# fatal under juliac --trim); a larger image-time constant is not that.  `AA` and `B` remain
# for `site_basis!`, which is `ace_site_basis`'s contract and is the only reader of the flat
# AA/A2B/WB representation.
mutable struct Workspace
    A::Vector{Float64}
    AA::Vector{Float64}
    B::Vector{Float64}
    ∂A::Vector{Float64}
    AAd::Vector{Float64}
    ∂AAd::Vector{Float64}
end

new_workspace() = Workspace(zeros(Float64, N_A), zeros(Float64, N_AA), zeros(Float64, N_BASIS),
                            zeros(Float64, N_A), zeros(Float64, N_DAG), zeros(Float64, N_DAG))
""")
    else
        println(io, """
mutable struct Workspace
    A::Vector{Float64}
    AA::Vector{Float64}
    B::Vector{Float64}
    ∂A::Vector{Float64}
    ∂AA::Vector{Float64}
end

new_workspace() = Workspace(zeros(Float64, N_A), zeros(Float64, N_AA), zeros(Float64, N_BASIS),
                            zeros(Float64, N_A), zeros(Float64, N_AA))
""")
    end

    # ---- E0 lookup -----------------------------------------------------------------------
    println(io, "# Reference energy of the centre species")
    println(io, "@inline function E0_of(iz0::Int)")
    _emit_species_dispatch(io, NZ, "    ", iz -> "return E0_$iz")
    println(io, "    error(\"E0_of: species index \$iz0 is outside 1:\$NZ\")")
    println(io, "end")
    println(io)

    # ---- per-pair A accumulation and force blocks ----------------------------------------
    println(io, """
# ============================================================================
# PER-PAIR A BLOCKS (the species-block accumulation)
# ============================================================================
#
# `_accA_k!` is the subset of ABASIS_SPEC that pair k's radial rows feed, written as straight
# line code over compile-time indices.  `_forceblk_k` is its transpose, grouped so that the
# per-edge work is the block's size and not N_RNL + N_YLM:
#
#   w[s] = Σ_{a in block, slot s} ∂A[a] * Y[y]     (this pair's rows of ∂Rnl)
#   g    = Σ_s w[s] * dR[s]                        (scalar; multiplied by r̂ by the caller)
#   u[y] = Σ_{a in block, harmonic y} ∂A[a] * R[s] (this pair's ∂Ylm)
#   v    = Σ_y u[y] * dY[y]                        (SVector{3})
#
# so each edge contributes ONE force vector and ONE rank-1 virial update, instead of one per
# (n,l) and one per lm.""")

    for k = 1:NZ^2
        blk = blocks[k]
        rows = pair_rows[k]
        m = length(rows)
        println(io, "\n# --- pair $k: $(length(blk)) of $nA A functions, $m radial rows ---")
        # forward
        println(io, "@inline function _accA_$(k)!(A::Vector{Float64}, R::SVector{M_RNL, Float64}, Y::SVector{N_YLM, Float64})")
        if isempty(blk)
            println(io, "    return nothing")
        else
            println(io, "    @inbounds begin")
            for (a, sl, y) in blk
                println(io, "        A[$a] += R[$sl] * Y[$y]")
            end
            println(io, "    end")
            println(io, "    return nothing")
        end
        println(io, "end")

        # backward
        println(io, "@inline function _forceblk_$k(∂A::Vector{Float64}, R::SVector{M_RNL, Float64}, " *
                    "dR::SVector{M_RNL, Float64}, Y::SVector{N_YLM, Float64}, " *
                    "dY::SVector{N_YLM, SVector{3, Float64}})")
        if isempty(blk)
            println(io, "    return 0.0, zero(SVector{3, Float64})")
        else
            slots = sort(unique(t[2] for t in blk))
            ylms  = sort(unique(t[3] for t in blk))
            println(io, "    @inbounds begin")
            for sl in slots
                println(io, "        w_$sl = 0.0")
                for (a, s2, y) in blk
                    s2 == sl && println(io, "        w_$sl += ∂A[$a] * Y[$y]")
                end
            end
            println(io, "        g = 0.0")
            for sl in slots
                println(io, "        g += w_$sl * dR[$sl]")
            end
            for y in ylms
                println(io, "        u_$y = 0.0")
                for (a, s2, y2) in blk
                    y2 == y && println(io, "        u_$y += ∂A[$a] * R[$s2]")
                end
            end
            println(io, "        v = zero(SVector{3, Float64})")
            for y in ylms
                println(io, "        v += u_$y * dY[$y]")
            end
            println(io, "        return g, v")
            println(io, "    end")
        end
        println(io, "end")
    end
    println(io)

    println(io, """
@inline function _accumulate_A_block!(A::Vector{Float64}, R::SVector{M_RNL, Float64},
                                      Y::SVector{N_YLM, Float64}, k::Int)""")
    _emit_pair_dispatch(io, NZ, "    ", k -> "return _accA_$(k)!(A, R, Y)")
    println(io, "    error(\"_accumulate_A_block!: species-pair index \$k is outside 1:\$(NZ*NZ)\")")
    println(io, "end")
    println(io)

    println(io, """
@inline function _force_block(∂A::Vector{Float64}, R::SVector{M_RNL, Float64},
                              dR::SVector{M_RNL, Float64}, Y::SVector{N_YLM, Float64},
                              dY::SVector{N_YLM, SVector{3, Float64}}, k::Int)""")
    _emit_pair_dispatch(io, NZ, "    ", k -> "return _forceblk_$k(∂A, R, dR, Y, dY)")
    println(io, "    error(\"_force_block: species-pair index \$k is outside 1:\$(NZ*NZ)\")")
    println(io, "end")
    println(io)

    # ---- aabasis forward -----------------------------------------------------------------
    aabasis = tensor.aabasis
    max_order = length(aabasis.specs)

    println(io, use_dag ? """
# ============================================================================
# TENSOR: A -> AA -> B   (trim-safe, manual)  --  `site_basis` ONLY
# ============================================================================
#
# With aa_products = :dag the flat AA -> B route is NOT on the energy/force path; that path is
# the DAG below.  It survives because `ace_site_basis` returns the N_BASIS descriptor vector
# `B`, which the DAG representation does not compute (the readout is folded into CTILDE, so
# there is no `B` left in it).

# Manual forward pass through SparseSymmProd (aabasis): A -> AA
@inline function evaluate_aabasis!(AA::Vector{Float64}, A::Vector{Float64})
""" : """
# ============================================================================
# TENSOR: A -> AA -> B  and its pullback  ∂B -> ∂AA -> ∂A   (trim-safe, manual)
# ============================================================================

# Manual forward pass through SparseSymmProd (aabasis): A -> AA
@inline function evaluate_aabasis!(AA::Vector{Float64}, A::Vector{Float64})
""")
    for ord in 1:max_order
        spec = aabasis.specs[ord]
        isempty(spec) && continue
        range_start = aabasis.ranges[ord].start
        range_stop = aabasis.ranges[ord].stop
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
            println(io, "        AA[i] = prod(A[ϕ[t]] for t in 1:$ord)")
        end
        println(io, "    end")
        println(io)
    end
    println(io, "    return AA")
    println(io, "end")
    println(io)

    # ---- aabasis pullback (:flat only) ----------------------------------------------------
    if !use_dag
        println(io, """
# Manual pullback through SparseSymmProd (aabasis): ∂AA -> ∂A
@inline function pullback_aabasis!(∂A::Vector{Float64}, ∂AA::Vector{Float64}, A::Vector{Float64})
""")
        for ord in 1:max_order
            spec = aabasis.specs[ord]
            isempty(spec) && continue
            range_start = aabasis.ranges[ord].start
            range_stop = aabasis.ranges[ord].stop
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

        println(io, """
# Static product with gradient (general-order fallback used by pullback_aabasis!)
@inline _static_prod_ed(b::NTuple{1, T}) where {T} = (b[1], (one(T),))
@inline _static_prod_ed(b::NTuple{2, T}) where {T} = (b[1] * b[2], (b[2], b[1]))
@inline function _static_prod_ed(b::NTuple{3, T}) where {T}
    p12 = b[1] * b[2]
    return p12 * b[3], (b[2] * b[3], b[1] * b[3], p12)
end
@inline function _static_prod_ed(b::NTuple{4, T}) where {T}
    p12 = b[1] * b[2]
    p34 = b[3] * b[4]
    return p12 * p34, (b[2] * p34, b[1] * p34, p12 * b[4], p12 * b[3])
end
""")
    end

    # ---- A2B ------------------------------------------------------------------------------
    println(io, use_dag ? """
# AA -> B  (sparse A2Bmap product) -- `site_basis` only, see above
@inline function _tensor_B!(ws::Workspace)""" : """
# AA -> B  (sparse A2Bmap product)
@inline function _tensor_B!(ws::Workspace)""")
    println(io, """
    evaluate_aabasis!(ws.AA, ws.A)
    B = ws.B
    fill!(B, 0.0)
    AA = ws.AA
    @inbounds for idx in eachindex(A2BMAP_1_I)
        B[A2BMAP_1_I[idx]] += A2BMAP_1_V[idx] * AA[A2BMAP_1_J[idx]]
    end
    return B
end

# ============================================================================
# PASS 1: per-neighbour embeddings, A accumulated inside the pair's block only
# ============================================================================
#
# Values only -- the derivative route needs no more than this, because pass 2 re-evaluates
# the edge (see the header).  Returns the pair-potential sum.
@inline function _embed_val!(ws::Workspace, Rs::AbstractVector{SVector{3, Float64}},
                             Zs::AbstractVector{<:Integer}, iz0::Int)
    A = ws.A
    fill!(A, 0.0)
    Epair = 0.0
    @inbounds for j in 1:length(Rs)
        Rj = Rs[j]
        r = norm(Rj)
        r <= 1e-10 && continue
        jz = z2i(Zs[j])
        k = pair_idx(iz0, jz)
        R = _evaluate_Rnl_pair_m(r, k)
        Y = eval_ylm(Rj)
        _accumulate_A_block!(A, R, Y, k)
        Epair += pair_energy(r, iz0, jz)
    end
    return Epair
end
""")

    # ---- the TENSOR STEP: :dag or :flat ----------------------------------------------------
    if use_dag
    println(io, """
# ============================================================================
# TENSOR STEP: the AA DAG, the folded readout, and ∂A for the force pass  (B3)
# ============================================================================
#
# WHAT B3 REMOVED.  Before it, one site cost: a flat AA pass (one product tree per AA
# function, sharing nothing between them); a sparse A2B product AA -> B; `dot(B, WB_iz)`; a
# SECOND traversal of the same sparse map to seed ∂AA from WB_iz -- with `A2Bmapᵀ · WB_iz` a
# PER-SPECIES CONSTANT recomputed at every site; and a flat pullback that re-multiplied the
# order-N products term by term.  B, the A2B map and WB are on none of those paths now:
#
#   forward   AAd[n] = AAd[n1] * AAd[n2]     ONE multiply per node, subproducts shared
#   energy    Ei = dot(CTILDE_iz, AAd)       the readout, folded at export time
#   backward  ∂AAd .= CTILDE_iz;  then per node  ∂AAd[n1] += w*AAd[n2]; ∂AAd[n2] += w*AAd[n1]
#
# i.e. TWO FMAs per node, seeded by the same constant vector the energy used.  Nothing is
# zeroed: `AAd` and `∂AAd` are fully overwritten, and `∂A` is written (not accumulated) from
# the leaf cotangents.
#
# The backward loop runs the nodes in DESCENDING index order.  That is a valid reverse
# topological order because every node's parents have a STRICTLY LARGER index than it (the
# builder only ever combines nodes that already exist), so by the time node i is read, every
# contribution to ∂AAd[i] has been made.

# Forward pass: A -> AAd (all N_DAG node values).
@inline function dag_forward!(AAd::Vector{Float64}, A::Vector{Float64})
    @inbounds begin""")
    if dag.has0
        println(io, "        AAd[1] = 1.0")
    end
    println(io, """        for i in 1:DAG_NUM1
            AAd[$(has0 == 0 ? "i" : "$has0 + i")] = A[i]
        end
        for j in eachindex(DAG_NODES)
            n1, n2 = DAG_NODES[j]
            AAd[DAG_FIRST - 1 + j] = AAd[n1] * AAd[n2]
        end
    end
    return AAd
end

# Energy only (no pullback): the entry `ace_site_energy` takes.  Shares dag_forward! and
# CTILDE with the force path, so `ace_site_energy` and `ace_site_energy_forces` agree BITWISE
# on the many-body term.
@inline function tensor_energy!(AAd::Vector{Float64}, A::Vector{Float64}, iz0::Int)
    dag_forward!(AAd, A)
    Ei = 0.0""")
    _emit_species_dispatch_multi(io, NZ, "    ", iz -> ["Ei = dot(CTILDE_$iz, AAd)"])
    println(io, """    return Ei
end

# Forward + readout + pullback to ∂A, in one pass down the DAG.  This is the signature Task
# 7's brief specifies; `_energy_and_∂A!` below is the workspace-shaped wrapper the entry
# points call.
@inline function tensor_energy_and_∂A!(∂A::Vector{Float64}, AAd::Vector{Float64},
                                       ∂AAd::Vector{Float64}, A::Vector{Float64}, iz0::Int)
    dag_forward!(AAd, A)
    Ei = 0.0""")
    _emit_species_dispatch_multi(io, NZ, "    ", iz -> [
        "Ei = dot(CTILDE_$iz, AAd)",
        "copyto!(∂AAd, CTILDE_$iz)",
    ])
    println(io, """    @inbounds for j in length(DAG_NODES):-1:1
        w = ∂AAd[DAG_FIRST - 1 + j]
        n1, n2 = DAG_NODES[j]
        ∂AAd[n1] = muladd(w, AAd[n2], ∂AAd[n1])
        ∂AAd[n2] = muladd(w, AAd[n1], ∂AAd[n2])
    end
    @inbounds for i in 1:DAG_NUM1
        ∂A[i] = ∂AAd[$(has0 == 0 ? "i" : "$has0 + i")]
    end""")
    if dag.num1 < nA
        println(io, """    # A functions $(dag.num1 + 1):$nA appear in no AA function, so their
    # cotangent is identically zero; ∂A is WRITTEN rather than accumulated, so say so.
    @inbounds for i in $(dag.num1 + 1):N_A
        ∂A[i] = 0.0
    end""")
    end
    println(io, """    return Ei
end

@inline _energy_and_∂A!(ws::Workspace, iz0::Int) =
    tensor_energy_and_∂A!(ws.∂A, ws.AAd, ws.∂AAd, ws.A, iz0)
""")
    else
        # The B2 tensor step, unchanged and bit-identical to b826c831's.  `∂Ei/∂B` IS the
        # readout weight vector, so the transposed A2B product is taken directly against WB
        # rather than through a ∂B copy.  Folding `A2Bmapᵀ · WB` into a per-species constant
        # is what aa_products = :dag does; it is deliberately NOT done here, so that this path
        # stays the exact expression, in the exact order, that every gate in Tasks 4-6 timed.
        println(io, """
# ============================================================================
# TENSOR STEP: energy readout, and ∂A for the force pass
# ============================================================================
@inline function _energy_and_∂A!(ws::Workspace, iz0::Int)
    B = _tensor_B!(ws)
    ∂AA = ws.∂AA
    fill!(∂AA, 0.0)
    Ei = 0.0""")
        _emit_species_dispatch_multi(io, NZ, "    ", iz -> [
            "Ei = dot(B, WB_$iz)",
            "@inbounds for idx in eachindex(A2BMAP_1_I)",
            "    ∂AA[A2BMAP_1_J[idx]] += A2BMAP_1_V[idx] * WB_$(iz)[A2BMAP_1_I[idx]]",
            "end",
        ])
        println(io, """
    ∂A = ws.∂A
    fill!(∂A, 0.0)
    pullback_aabasis!(∂A, ∂AA, ws.A)
    return Ei
end
""")
    end

    println(io, """
# ============================================================================
# PASS 2: forces (and virial) from ∂A
# ============================================================================
#
# One force vector and ONE rank-1 virial update per EDGE.  Before B2 the virial was updated
# once per (n,l) and once per lm inside the force loop: 78 outer products per edge on Cantor,
# which is the same number in exact arithmetic and 78x the work.
@inline function _forces_from_∂A!(forces::AbstractVector{SVector{3, Float64}}, ws::Workspace,
                                  Rs::AbstractVector{SVector{3, Float64}},
                                  Zs::AbstractVector{<:Integer}, iz0::Int,
                                  with_virial::Bool)
    ∂A = ws.∂A
    Epair = 0.0
    vir = zero(SMatrix{3, 3, Float64, 9})
    @inbounds for j in 1:length(Rs)
        Rj = Rs[j]
        r = norm(Rj)
        if r <= 1e-10
            forces[j] = zero(SVector{3, Float64})
            continue
        end
        jz = z2i(Zs[j])
        k = pair_idx(iz0, jz)
        R, dR = _evaluate_Rnl_d_pair_m(r, k)
        Y, dY = eval_ylm_ed(Rj)
        g, v = _force_block(∂A, R, dR, Y, dY, k)
        # Pair potential (ordered pair: centre species first)
        ep, dep = pair_energy_d(r, iz0, jz)
        Epair += ep
        f = (g + dep) * (Rj / r) + v
        forces[j] = -f          # force is the negative gradient
        if with_virial
            vir = vir - Rj * f'
        end
    end
    return Epair, vir
end

# ============================================================================
# ENTRY POINTS
# ============================================================================
#
# The `!` forms take the workspace; the positional forms allocate one and are what the tests,
# the diagnostic scripts and `_write_main` call.  Both compute the same expression in the same
# order, so they agree bitwise.

function site_energy!(ws::Workspace, Rs::AbstractVector{SVector{3, Float64}},
                      Zs::AbstractVector{<:Integer}, Z0::Integer)
    iz0 = z2i(Z0)
    length(Rs) == 0 && return E0_of(iz0)
    Epair = _embed_val!(ws, Rs, Zs, iz0)""")
    if use_dag
        println(io, """    Emb = tensor_energy!(ws.AAd, ws.A, iz0)
    return (Emb + Epair) + E0_of(iz0)
end""")
    else
        println(io, """    B = _tensor_B!(ws)
    Emb = 0.0""")
        _emit_species_dispatch_multi(io, NZ, "    ", iz -> ["Emb = dot(B, WB_$iz)"])
        println(io, """    return (Emb + Epair) + E0_of(iz0)
end""")
    end
    println(io, """

function site_energy_forces_virial!(ws::Workspace, Rs::AbstractVector{SVector{3, Float64}},
                                    Zs::AbstractVector{<:Integer}, Z0::Integer,
                                    forces::AbstractVector{SVector{3, Float64}})
    iz0 = z2i(Z0)
    if length(Rs) == 0
        return E0_of(iz0), zero(SMatrix{3, 3, Float64, 9})
    end
    _embed_val!(ws, Rs, Zs, iz0)
    Emb = _energy_and_∂A!(ws, iz0)
    Epair, vir = _forces_from_∂A!(forces, ws, Rs, Zs, iz0, true)
    return (Emb + Epair) + E0_of(iz0), vir
end

function site_energy_forces!(ws::Workspace, Rs::AbstractVector{SVector{3, Float64}},
                             Zs::AbstractVector{<:Integer}, Z0::Integer,
                             forces::AbstractVector{SVector{3, Float64}})
    iz0 = z2i(Z0)
    length(Rs) == 0 && return E0_of(iz0)
    _embed_val!(ws, Rs, Zs, iz0)
    Emb = _energy_and_∂A!(ws, iz0)
    Epair, _ = _forces_from_∂A!(forces, ws, Rs, Zs, iz0, false)
    return (Emb + Epair) + E0_of(iz0)
end

function site_basis!(ws::Workspace, Rs::AbstractVector{SVector{3, Float64}},
                     Zs::AbstractVector{<:Integer}, Z0::Integer)
    length(Rs) == 0 && return zeros(Float64, N_BASIS)
    _embed_val!(ws, Rs, Zs, z2i(Z0))
    return copy(_tensor_B!(ws))
end

site_energy(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer) =
    site_energy!(new_workspace(), Rs, Zs, Z0)

site_basis(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer) =
    site_basis!(new_workspace(), Rs, Zs, Z0)

function site_energy_forces(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    forces = Vector{SVector{3, Float64}}(undef, length(Rs))
    E = site_energy_forces!(new_workspace(), Rs, Zs, Z0, forces)
    return E, forces
end

function site_energy_forces_virial(Rs::Vector{SVector{3, Float64}}, Zs::Vector{<:Integer}, Z0::Integer)
    forces = Vector{SVector{3, Float64}}(undef, length(Rs))
    E, V = site_energy_forces_virial!(new_workspace(), Rs, Zs, Z0, forces)
    return E, forces, V
end

# ============================================================================
# DIAGNOSTIC EMBEDDINGS (full width, allocating -- NOT on the hot path)
# ============================================================================
#
# These reproduce the pre-B2 `compute_embeddings*` signatures at full N_RNL width, which is
# what export/test/test_pair_export.jl uses to attribute the value-route / derivative-route
# disagreement to the radial half or the spherical-harmonic half.  They allocate their own
# output, so (unlike the pre-B2 versions, which shared one set of global scratch arrays)
# the value-route result does NOT have to be copied before the derivative route runs.

function compute_embeddings(Rs::AbstractVector{SVector{3, Float64}},
                            Zs::AbstractVector{<:Integer}, Z0::Integer)
    nneigh = length(Rs)
    iz0 = z2i(Z0)
    Rnl = zeros(Float64, nneigh, N_RNL)
    Ylm = zeros(Float64, nneigh, N_YLM)
    for j in 1:nneigh
        r = norm(Rs[j])
        r <= 1e-10 && continue
        Rj = evaluate_Rnl(r, iz0, z2i(Zs[j]))
        Yj = eval_ylm(Rs[j])
        for t in 1:N_RNL; Rnl[j, t] = Rj[t]; end
        for t in 1:N_YLM; Ylm[j, t] = Yj[t]; end
    end
    return Rnl, Ylm
end

function compute_embeddings_ed(Rs::AbstractVector{SVector{3, Float64}},
                               Zs::AbstractVector{<:Integer}, Z0::Integer)
    nneigh = length(Rs)
    iz0 = z2i(Z0)
    Rnl = zeros(Float64, nneigh, N_RNL)
    dRnl = zeros(Float64, nneigh, N_RNL)
    Ylm = zeros(Float64, nneigh, N_YLM)
    dYlm = fill(zero(SVector{3, Float64}), nneigh, N_YLM)
    rs = zeros(Float64, nneigh)
    rhats = fill(zero(SVector{3, Float64}), nneigh)
    for j in 1:nneigh
        r = norm(Rs[j])
        rs[j] = r
        r <= 1e-10 && continue
        rhats[j] = Rs[j] / r
        Rj, dRj = evaluate_Rnl_d(r, iz0, z2i(Zs[j]))
        Yj, dYj = eval_ylm_ed(Rs[j])
        for t in 1:N_RNL; Rnl[j, t] = Rj[t]; dRnl[j, t] = dRj[t]; end
        for t in 1:N_YLM; Ylm[j, t] = Yj[t]; dYlm[j, t] = dYj[t]; end
    end
    return Rnl, dRnl, Ylm, dYlm, rs, rhats
end
""")
end
