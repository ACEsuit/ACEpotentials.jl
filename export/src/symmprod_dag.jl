# Binary DAG for the AA (symmetric-product) basis -- EXPORT TIME ONLY.
#
# ============================================================================================
# WHY THIS FILE EXISTS INSTEAD OF `EquivariantTensors.SparseSymmProdDAG`
# ============================================================================================
#
# Task 7's brief says to consume `EquivariantTensors.SparseSymmProdDAG` from
# `EquivariantTensors/src/ace/symmprod_dag.jl`.  That type DOES NOT EXIST at runtime, and it
# is not a version skew -- it does not exist in ANY of the seven EquivariantTensors versions
# installed in this depot:
#
#   * `src/ace/symmprod_dag.jl` is NOT in the `include` list of `src/EquivariantTensors.jl`
#     (check: `grep -n include src/EquivariantTensors.jl`).  Its first line reads
#     "# TO BE RE-INTEGRATED INTO CODEBASE".
#   * It could not be `include`d even by hand: the struct is declared
#     `<: AbstractP4MLTensor` with an `@reqfields` member and a `_make_reqfields()`
#     constructor, and none of those three names exists in EquivariantTensors OR in the
#     pinned Polynomials4ML (v0.5.8) any more.  They are a removed P4ML API.
#   * `isdefined(EquivariantTensors, :SparseSymmProdDAG) == false` is the one-line check.
#
# The plan forbids modifying EquivariantTensors, so the DAG CONSTRUCTION is ported here,
# into `export/`, where all of this task's work is required to live.  The port is faithful:
# `_score_partition`, `_get_ns`, `_find_partition`, `_insert_partition!` and the constructor
# body are line-for-line the upstream algorithm, and the node/leaf/`has0` layout and the
# `projection` semantics are upstream's exactly, so the evaluation kernels in
# `symmprod_dag_kernels.jl` (`evaluate!`, `unsafe_pullback!`) describe what the GENERATED code
# does.  Two deliberate departures, both narrowing:
#
#   1. `Combinatorics.partitions(1:n)` is replaced by `_set_partitions(n)` below, so that
#      `export/` gains no new dependency and the enumeration ORDER -- which decides ties in
#      the partition score, and therefore the exact DAG -- is pinned in this repository
#      rather than in a transitive dependency's iteration order.
#   2. A `kk` that is ALREADY a node (an intermediate product inserted for an earlier basis
#      function happens to have exactly this multiset) is reused instead of re-inserted.
#      Upstream would pick the one-block partition here, score it best, and then index
#      `p[2]` out of a length-1 vector.  See `_insert_partition!`.
#
# NOTHING IN THIS FILE RUNS IN A GENERATED LIBRARY.  It runs in the generator; what reaches
# the library is `DAG_NODES`, `N_DAG` and the folded `CTILDE_*` vectors.
#
# ============================================================================================
# THE LAYOUT (upstream's, and what the generated forward/backward passes assume)
# ============================================================================================
#
#   node 1                        : the constant, iff `has0`          (value 1.0)
#   nodes has0+1 .. has0+num1     : LEAVES, `AAd[has0+i] = A[i]`.  Upstream stores `(i+has0, 0)`
#                                   in `nodes[has0+i]`, i.e. a self-reference; the evaluator
#                                   ignores it and copies `A` positionally.  We emit nothing
#                                   for these -- `DAG_NODES` holds only the interior nodes.
#   nodes has0+num1+1 .. N_DAG    : INTERIOR, `AAd[i] = AAd[n1] * AAd[n2]`, with n1, n2 < i.
#
# `num1` is the LARGEST A index that appears anywhere in the spec, not the number of order-1
# AA functions: the order-1 AA basis is usually a strict subset of A, but every A index used
# by any higher-order function still needs a leaf.  If `num1 < N_A` then `∂A[num1+1:N_A]` is
# identically zero and the generated code zeroes it explicitly.
#
# `projection[k]` is the DAG node holding flat AA function `k`, in the flat AA ordering
# (`SparseSymmProd`'s: the constant first if `hasconst`, then order 1, 2, ... in `ranges`).

const BinDagNode = Tuple{Int, Int}

"""
    SymmProdDAG

Binary DAG evaluating an AA basis.  Fields as upstream `SparseSymmProdDAG`, minus the
allocation pools that type carries: `nodes`, `has0`, `num1`, `projection`.
"""
struct SymmProdDAG
    nodes::Vector{BinDagNode}
    has0::Bool
    num1::Int
    projection::Vector{Int}
end

Base.length(dag::SymmProdDAG) = length(dag.nodes)

"""
    _set_partitions(n) -> Vector{Vector{Vector{Int}}}

Every set partition of `1:n`, each block sorted ascending and the blocks ordered by their
smallest element.  Generated from restricted growth strings, so the enumeration order is
fixed by this function and not by any dependency.  `length(_set_partitions(n))` is the Bell
number `B_n` (1, 2, 5, 15, 52, 203 for n = 1..6), which `export/test/test_dag.jl` asserts.

Replaces `Combinatorics.partitions(1:n)`; `n` here is a correlation order (3 or 4 on the
benchmark models), so the cost is irrelevant.
"""
function _set_partitions(n::Int)
    n <= 0 && return [Vector{Int}[]]
    out = Vector{Vector{Vector{Int}}}()
    a = zeros(Int, n)          # restricted growth string, a[1] == 1
    function rec(i::Int, mx::Int)
        if i > n
            nb = mx
            blocks = [Int[] for _ in 1:nb]
            for j in 1:n
                push!(blocks[a[j]], j)
            end
            push!(out, blocks)
            return
        end
        for v in 1:(mx + 1)
            a[i] = v
            rec(i + 1, max(mx, v))
        end
    end
    rec(1, 0)
    return out
end

# --- the upstream partition heuristic, verbatim -----------------------------------------

_score_partition(p) = isempty(p) ? Inf : (1e9 * length(p) + maximum(p))

function _get_ns(p, specnew_dict)
    out = Vector{Int}(undef, length(p))
    for (i, kk_) in enumerate(p)
        if haskey(specnew_dict, kk_)
            out[i] = specnew_dict[kk_]
        else
            return Int[]
        end
    end
    return out
end

function _find_partition(kk, specnew_dict, has0::Bool)
    worstp = _get_ns([[k] for k in kk], specnew_dict)
    @assert worstp == has0 .+ kk
    bestp = worstp
    bestscore = _score_partition(bestp)

    for ip in _set_partitions(length(kk))
        p = _get_ns([sort(kk[i]) for i in ip], specnew_dict)
        score = _score_partition(p)
        if !isempty(p) && score < bestscore
            bestp = p
            bestscore = score
        end
    end

    return bestp
end

# Returns the number of EXTRA (auxiliary) nodes inserted.  `nodes` and `specnew` are grown in
# lockstep, so a node's index into `nodes` is also its index into `specnew` -- which is what
# makes `length(nodes)` a usable node reference in the recursion below.
function _insert_partition!(nodes, specnew, specnew_dict, kk, p)
    @assert length(p) >= 2 """
        _insert_partition!: partition of $kk has $(length(p)) block(s); a 1-block partition
        means kk is already a node and must be handled by the caller"""
    if length(p) == 2
        push!(nodes, BinDagNode((p[1], p[2])))
        push!(specnew, kk)
        specnew_dict[kk] = length(specnew)
        return 0
    else
        push!(nodes, BinDagNode((p[1], p[2])))
        kk1 = sort(vcat(specnew[p[1]], specnew[p[2]]))
        push!(specnew, kk1)
        specnew_dict[kk1] = length(specnew)
        return 1 + _insert_partition!(nodes, specnew, specnew_dict,
                                      kk, vcat([length(nodes)], p[3:end]))
    end
end

"""
    SymmProdDAG(spec::AbstractVector{<:AbstractVector{<:Integer}}) -> SymmProdDAG

`spec` is the FLAT AA specification in flat AA order: a list of sorted index vectors, sorted
by length, optionally starting with the empty vector (the constant).  Use
`aa_flat_spec(aabasis)`.
"""
function SymmProdDAG(spec::AbstractVector)
    @assert issorted(length.(spec)) "AA spec must be sorted by correlation order"
    @assert all(issorted, spec) "every AA spec entry must be sorted"

    has0 = (length(spec[1]) == 0)
    spec1 = spec[length.(spec) .== 1]
    IN = (length(spec1) + 1 + has0):length(spec)
    specN = spec[IN]

    nodes = BinDagNode[]
    sizehint!(nodes, length(spec))
    specnew = Vector{Int}[]
    specnew_dict = Dict{Vector{Int}, Int}()
    sizehint!(specnew, length(spec))

    if has0
        push!(nodes, BinDagNode((0, 0)))
        push!(specnew, Int[])
        specnew_dict[Int[]] = length(specnew)
    end

    # The full 1-particle basis, up to the largest A index used anywhere in the spec.
    _mymax(vv) = length(vv) == 0 ? 0 : maximum(vv)
    num1 = maximum(_mymax(vv) for vv in spec)
    for i = 1:num1
        push!(nodes, BinDagNode((i + has0, 0)))
        push!(specnew, [i])
        specnew_dict[[i]] = length(specnew)
    end

    for kk in specN
        kkv = collect(Int, kk)
        # Already a node?  (An auxiliary node inserted for an earlier basis function can carry
        # exactly this multiset.)  Then there is nothing to insert and `projection` will find
        # it.  Upstream crashes here instead; see the header.
        haskey(specnew_dict, kkv) && continue
        p = _find_partition(kkv, specnew_dict, has0)
        _insert_partition!(nodes, specnew, specnew_dict, kkv, p)
    end

    projection = [specnew_dict[collect(Int, vv)] for vv in spec]

    return SymmProdDAG(nodes, has0, num1, projection)
end

"""
    aa_flat_spec(aabasis) -> Vector{Vector{Int}}

The flat AA specification of an `EquivariantTensors.SparseSymmProd`, in the order its
`evaluate!` writes: the empty vector first iff `hasconst`, then `specs[1]`, `specs[2]`, ...
Same content as `EquivariantTensors.reconstruct_spec(aabasis)`, written out here so the
generator does not depend on an unexported upstream helper.
"""
function aa_flat_spec(aabasis)
    spec = Vector{Int}[]
    aabasis.hasconst && push!(spec, Int[])
    for s in aabasis.specs, bb in s
        push!(spec, sort(collect(Int, bb)))
    end
    return spec
end

"""
    reconstruct_dag_spec(dag) -> Vector{Vector{Int}}

The A-index multiset each DAG node computes, derived from the node list ALONE (no reference
to the spec the DAG was built from).  `export/test/test_dag.jl` uses it as a structural check
that is independent of the numeric gates: node `projection[k]` must carry exactly flat AA
function `k`'s multiset.
"""
function reconstruct_dag_spec(dag::SymmProdDAG)
    spec = Vector{Int}[]
    has0 = dag.has0
    for i = 1:length(dag.nodes)
        n1, n2 = dag.nodes[i]
        if n1 == n2 == 0
            push!(spec, Int[])
        elseif n2 == 0
            push!(spec, Int[n1 - has0])
        else
            push!(spec, sort(vcat(spec[n1], spec[n2])))
        end
    end
    return spec
end

"""
    evaluate_dag(dag, A) -> AAd

Reference forward pass, identical in structure to the generated one and to upstream's
`evaluate!(AA, ::SparseSymmProdDAG, A)`.  Test-side only.
"""
function evaluate_dag(dag::SymmProdDAG, A::AbstractVector{T}) where {T}
    AAd = zeros(T, length(dag.nodes))
    has0 = dag.has0
    has0 && (AAd[1] = one(T))
    @inbounds for i = 1:dag.num1
        AAd[has0 + i] = A[i]
    end
    @inbounds for i = (dag.num1 + has0 + 1):length(dag.nodes)
        n1, n2 = dag.nodes[i]
        AAd[i] = AAd[n1] * AAd[n2]
    end
    return AAd
end

"""
    pullback_dag(dag, ct, AAd) -> ∂A

Reference BACKWARD pass, written to mirror `tensor_energy_and_∂A!`'s emitted loops statement
for statement — seeded by the same `ct` the energy used, two FMAs per node, descending node
order, and the same `has0` leaf offset.  Test-side only; its purpose is that
`export/test/test_dag.jl` can exercise the `has0 = true` offsets, which no model in this
repository reaches (see `_dag_leaf_ix` in `write_evaluation.jl`).
"""
function pullback_dag(dag::SymmProdDAG, ct::AbstractVector{T}, AAd::AbstractVector{T}) where {T}
    n = length(dag.nodes)
    has0 = dag.has0 ? 1 : 0
    ∂AAd = collect(T, ct)
    @inbounds for i = n:-1:(has0 + dag.num1 + 1)
        w = ∂AAd[i]
        n1, n2 = dag.nodes[i]
        ∂AAd[n1] = muladd(w, AAd[n2], ∂AAd[n1])
        ∂AAd[n2] = muladd(w, AAd[n1], ∂AAd[n2])
    end
    return [∂AAd[has0 + i] for i = 1:dag.num1]
end

"""
    dag_ctilde(dag, ct_flat) -> Vector{Float64}

Scatter a flat readout covector (`A2Bmap' * WB_iz`, one entry per FLAT AA function) onto the
DAG's nodes, so that `dot(ctilde, AAd) == dot(ct_flat, AA_flat)`.  `projection` is injective
(distinct AA functions are distinct multisets, and `specnew_dict` is keyed on the multiset),
so this is a permutation-scatter into a longer, mostly-zero vector; `+=` is used anyway so a
non-injective projection would sum rather than silently drop.
"""
function dag_ctilde(dag::SymmProdDAG, ct_flat::AbstractVector)
    ct = zeros(Float64, length(dag.nodes))
    @assert length(ct_flat) == length(dag.projection)
    for (k, n) in enumerate(dag.projection)
        ct[n] += ct_flat[k]
    end
    return ct
end
