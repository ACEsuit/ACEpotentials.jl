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

@testset "exported DAG evaluator is exact" begin
    fx = load_cantor_fixture()
    f = joinpath(DAG_BUILD, "cantor_dag.jl")
    Base.invokelatest(export_ace_model, fx.stacked, f; radial_basis = :polynomial)
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
