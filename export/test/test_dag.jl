#=
The AA product DAG (Task 7 / B3).

WHAT THIS FILE GATES, and why it is more than the numeric export gates already are.

`export/src/symmprod_dag.jl` is a PORT.  The brief's plan was to consume
`EquivariantTensors.SparseSymmProdDAG`; that type is not loadable (its file is not in the
module's include list, and it is declared against an `@reqfields` / `AbstractP4MLTensor` API
that no installed Polynomials4ML still has -- `isdefined(EquivariantTensors,
:SparseSymmProdDAG) == false`).  So there is no upstream implementation to diff against, and
the usual "it agrees with ET" argument is unavailable.  What replaces it here:

  1. `_set_partitions` is checked against the Bell numbers and against the definition of a set
     partition, because it decides the shape of every DAG this generator builds.
  2. STRUCTURAL: each node's A-index multiset is reconstructed FROM THE NODE LIST ALONE and
     compared with the flat AA specification through `projection`.  This is independent of
     any numeric evaluation: a DAG that multiplied the right number of right-looking factors
     in the wrong combination would pass a single random-vector check with probability zero
     but would be caught here by construction, and the failure would NAME the basis function.
  3. NUMERIC: the DAG's node values reproduce the flat AA basis, and the folded `CTILDE`
     reproduces `dot(A2Bmapᵀ WB, AA_flat)` -- on both benchmark models, every species.
  4. END TO END: the generated file no longer reads `A2BMAP` / `WB` on the energy/force path,
     still exports them for `site_basis`, and reproduces the Julia calculator at 1e-12.

Run standalone:  cd export/test && julia --project=.. -e 'include("test_dag.jl")'
=#

using Test, LinearAlgebra, SparseArrays, Random

include(joinpath(@__DIR__, "check_export.jl"))                       # + cantor fixture
include(joinpath(@__DIR__, "fixtures", "tial_fixture.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))   # brings symmprod_dag.jl

const DAG_BUILD = mkpath(joinpath(@__DIR__, "build"))

"Reference Bell numbers B_1 .. B_7."
const BELL = [1, 2, 5, 15, 52, 203, 877]

@testset "set partitions" begin
    for n in 1:7
        ps = _set_partitions(n)
        @test length(ps) == BELL[n]
        # every partition is a partition: blocks disjoint, union 1:n, each block sorted
        for p in ps
            @test sort(vcat(p...)) == collect(1:n)
            @test all(issorted, p)
            @test all(!isempty, p)
        end
        @test length(unique(map(p -> sort(p), ps))) == BELL[n]   # no duplicates
    end
end

"""
    dag_checks(name, tensor, ps_readout_W)

Structural + numeric checks of the DAG built for one model's AA basis.  Returns the DAG so a
caller can report its shape.
"""
function dag_checks(name, tensor, W)
    aab = tensor.aabasis
    spec = aa_flat_spec(aab)
    dag = SymmProdDAG(spec)
    nodes, num1, has0 = dag.nodes, dag.num1, (dag.has0 ? 1 : 0)

    @testset "$name: DAG shape" begin
        # leaves carry a self-reference and a 0 second slot; interior nodes point strictly
        # BACKWARDS, which is what makes one ascending and one descending pass correct.
        for i in 1:num1
            @test nodes[has0 + i] == (Int(has0 + i), 0)
        end
        has0 == 1 && @test nodes[1] == (0, 0)
        for i in (has0 + num1 + 1):length(nodes)
            n1, n2 = nodes[i]
            @test 1 <= n1 < i
            @test 1 <= n2 < i
        end
        @test num1 == maximum(maximum(s) for s in spec if !isempty(s))
        @test num1 <= length(tensor.abasis)
        @test length(dag.projection) == length(spec)
        @test length(unique(dag.projection)) == length(spec)   # injective
    end

    @testset "$name: structural (A-multiset per node)" begin
        rspec = reconstruct_dag_spec(dag)
        @test length(rspec) == length(nodes)
        bad = [k for k in eachindex(spec) if rspec[dag.projection[k]] != collect(Int, spec[k])]
        @test isempty(bad)
        isempty(bad) || @info "$name: first mismatching AA function" k=bad[1] want=spec[bad[1]] got=rspec[dag.projection[bad[1]]]
        # and every INTERIOR node's multiset is exactly the union of its two children's
        for i in (has0 + num1 + 1):length(nodes)
            n1, n2 = nodes[i]
            @test rspec[i] == sort(vcat(rspec[n1], rspec[n2]))
        end
    end

    @testset "$name: numeric (AA and the folded readout)" begin
        A2B = tensor.A2Bmaps[1]
        rng = MersenneTwister(3)
        for trial in 1:5
            A = randn(rng, length(tensor.abasis))
            AA_flat = ET.evaluate(aab, A)
            AAd = evaluate_dag(dag, A)
            @test maximum(abs.(AA_flat .- AAd[dag.projection])) <=
                  1e-13 * maximum(abs.(AA_flat))
            for iz in 1:size(W, 3)
                ct_flat = A2B' * W[1, :, iz]
                E_flat = dot(ct_flat, AA_flat)
                ct = dag_ctilde(dag, ct_flat)
                @test length(ct) == length(nodes)
                @test abs(dot(ct, AAd) - E_flat) <= 1e-12 * max(abs(E_flat), 1.0)
            end
        end
    end
    return dag
end

@testset "DAG reproduces the flat AA basis and the readout" begin
    fx = load_cantor_fixture()
    _, _, ace = cantor_substacks(fx)
    dc = dag_checks("cantor", ace.model.basis, ace.ps.readout.W)
    @info "cantor DAG" nodes=length(dc.nodes) leaves=dc.num1 flat_AA=length(ace.model.basis.aabasis)

    ft = load_tial_fixture()
    _, _, acet = tial_substacks(ft)
    dt = dag_checks("tial", acet.model.basis, acet.ps.readout.W)
    @info "tial DAG" nodes=length(dt.nodes) leaves=dt.num1 flat_AA=length(acet.model.basis.aabasis)
end

"""
Flat AA values and the flat pullback, for an arbitrary spec (`Int[]` allowed).  Independent of
`symmprod_dag.jl` -- it is the thing the DAG has to reproduce.
"""
function flat_ref(spec, A::Vector{T}, ct::Vector{T}) where {T}
    AA = [isempty(b) ? one(T) : prod(A[i] for i in b) for b in spec]
    E = sum(ct[k] * AA[k] for k in eachindex(spec))
    ∂A = zeros(T, length(A))
    for (k, b) in enumerate(spec), (t, i) in enumerate(b)
        p = ct[k]
        for (u, j) in enumerate(b); u == t || (p *= A[j]); end
        ∂A[i] += p
    end
    return AA, E, ∂A
end

@testset "has0 (a CONSTANT AA term): the paths no real model reaches" begin
    # WHY THIS EXISTS.  `SparseSymmProd.hasconst` is true only when the AA specification
    # contains the empty tuple, and no ACEpotentials model construction emits a 0-correlation
    # AA term -- so both benchmark models, and every model in this suite, are `hasconst =
    # false`.  The DAG's constant node, its `+has0` leaf offset and `reconstruct_dag_spec`'s
    # `n1 - has0` therefore have NO coverage from anything else here, and a future model with
    # a constant term would exercise an off-by-one with nothing to notice.
    #
    # WHAT THIS DOES AND DOES NOT COVER.  It covers the BUILDER and the reference kernels --
    # which are written to mirror the emitted loops statement for statement -- plus the one
    # `has0`-dependent index expression the generator emits (`_dag_leaf_ix`).  It does NOT
    # cover a compiled library with a constant AA term, because no model in this repository
    # can produce one; that remains uncovered and is stated as such in the task report.

    # the emitted index expression, both values
    @test _dag_leaf_ix(0) == "i"
    @test _dag_leaf_ix(1) == "1 + i"

    rng = MersenneTwister(11)
    for (label, spec) in (
        ("hasconst", [Int[], [1], [2], [3], [4], [1,1], [1,2], [2,3], [3,4], [4,4],
                      [1,1,2], [1,2,3], [2,3,4], [1,2,3,4], [2,2,3,3]]),
        ("no const",         [[1], [2], [3], [4], [1,1], [1,2], [2,3], [3,4], [4,4],
                      [1,1,2], [1,2,3], [2,3,4], [1,2,3,4], [2,2,3,3]]))
        dag = SymmProdDAG(spec)
        has0 = dag.has0 ? 1 : 0
        @test dag.has0 == (label == "hasconst")
        @test dag.num1 == 4
        @testset "$label: layout" begin
            has0 == 1 && @test dag.nodes[1] == (0, 0)
            for i in 1:dag.num1
                @test dag.nodes[has0 + i] == (has0 + i, 0)
            end
            for i in (has0 + dag.num1 + 1):length(dag.nodes)
                n1, n2 = dag.nodes[i]
                @test 1 <= n1 < i
                @test 1 <= n2 < i
                # the constant node is never a parent: partition blocks are non-empty, so no
                # node is ever built out of the empty multiset.
                has0 == 1 && @test n1 != 1 && n2 != 1
            end
        end
        @testset "$label: reconstruct_dag_spec (the n1 - has0 path)" begin
            rspec = reconstruct_dag_spec(dag)
            has0 == 1 && @test rspec[1] == Int[]
            for i in 1:dag.num1
                @test rspec[has0 + i] == [i]        # the off-by-one this guards
            end
            for k in eachindex(spec)
                @test rspec[dag.projection[k]] == spec[k]
            end
        end
        @testset "$label: forward, energy and pullback" begin
            for _ in 1:20
                A = randn(rng, 4)
                ct_flat = randn(rng, length(spec))
                AA, E, ∂A = flat_ref(spec, A, ct_flat)
                AAd = evaluate_dag(dag, A)
                has0 == 1 && @test AAd[1] == 1.0
                for i in 1:dag.num1
                    @test AAd[has0 + i] == A[i]     # the leaf offset _dag_leaf_ix emits
                end
                @test maximum(abs.(AA .- AAd[dag.projection])) <= 1e-12 * maximum(abs.(AA))
                ct = dag_ctilde(dag, ct_flat)
                @test abs(dot(ct, AAd) - E) <= 1e-12 * max(abs(E), 1.0)
                ∂A_dag = pullback_dag(dag, ct, AAd)
                @test maximum(abs.(∂A_dag .- ∂A)) <= 1e-12 * max(maximum(abs.(∂A)), 1.0)
            end
        end
    end
end

@testset "aa_products = :flat is the default and emits no DAG" begin
    fx = load_cantor_fixture()
    f = joinpath(DAG_BUILD, "cantor_flat_default.jl")
    Base.invokelatest(export_ace_model, fx.stacked, f; radial_basis = :polynomial)
    src = read(f, String)
    # The DEFAULT must be the gated, faster :flat tensor step (Task 7 measured :dag at 1.32x
    # SLOWER on this model end to end, even though its tensor step is 1.6x faster).
    @test !occursin("const DAG_NODES", src)
    @test !occursin("const CTILDE_1", src)
    @test occursin("pullback_aabasis!", src)
    @test occursin("WB_1", src)
    @test_throws ErrorException Base.invokelatest(export_ace_model, fx.stacked, f;
                                                  aa_products = :nonsense)
end

@testset "exported DAG evaluator is exact" begin
    fx = load_cantor_fixture()
    f = joinpath(DAG_BUILD, "cantor_dag.jl")
    Base.invokelatest(export_ace_model, fx.stacked, f;
                      radial_basis = :polynomial, aa_products = :dag)
    src = read(f, String)

    # the DAG constants are there ...
    @test occursin("const DAG_NODES", src)
    @test occursin("const N_DAG", src)
    @test occursin("const CTILDE_1", src)
    # ... the readout is folded: no WB and no A2B map on the energy/force path.  `A2BMAP_1_I`
    # and `WB_1` still EXIST (site_basis's contract), so the check is on the kernel, not on
    # the constants: `_tensor_B!` is the only reader of A2BMAP, and it is called only by
    # `site_basis!`.
    @test occursin("_tensor_B!", src)
    @test count(r"_tensor_B!\(", src) == 2        # one definition, one call (site_basis!)
    @test !occursin("pullback_aabasis!", src)
    @test occursin("tensor_energy_and_∂A!", src)
    for fn in ("_energy_and_∂A!", "tensor_energy!")
        @test occursin(fn, src)
    end

    dE, dF, dV = check_export_report(f, fx.stacked, fx.held, fx.rcut; label = "DAG export")
    @test dE <= 1e-12
    @test dF <= 1e-12
    @test dV <= 1e-12
end
