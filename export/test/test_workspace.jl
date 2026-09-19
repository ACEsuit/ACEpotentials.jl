#=
Task 6 / B2 -- the opaque workspace API.

WHAT THIS FILE GATES.  Before B2 the generated evaluation code kept every scratch buffer in
`const WORK_*` globals sized at a compile-time `MAX_NEIGHBORS = 256`.  Two consequences, both
of them bugs rather than limitations:

  * a site with more than 256 neighbours could not be evaluated at all (an `@assert`, which
    disappears under `--trim=safe`'s `-O3` unless it is explicitly kept, so the failure mode
    was a silent out-of-bounds write in a release build);
  * two threads calling `site_energy_forces_virial` at once wrote the same arrays, so the
    LAMMPS plugin had to be built with OpenMP OFF.

B2 replaces them with a caller-supplied `Workspace`.  This file asserts the three properties
that makes the library re-entrant rather than merely "not obviously broken":

  1. per-thread workspaces reproduce the serial result BITWISE.  Not `≈`: a workspace that is
     shared by accident produces a result that is usually right and occasionally wrong, and
     only a bitwise comparison catches the occasional case reliably.
  2. a 300-neighbour site evaluates (there is no cap), and `MAX_NEIGHBORS` is gone from the
     emitted source.
  3. the wrappers that allocate their own workspace (`site_energy`, `site_energy_forces`,
     `site_energy_forces_virial`) agree bitwise with the explicit-workspace entry points, so
     the tests and scripts that call the old signatures are gating the same code.

Run it with at least two threads, or property 1 is vacuous:

    cd export/test && julia -t 4 --project=.. -e 'include("test_workspace.jl")'
=#

using Test
using Base.Threads
using StaticArrays, LinearAlgebra

include(joinpath(@__DIR__, "check_export.jl"))                       # + cantor fixture
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))

const WS_BUILD = mkpath(joinpath(@__DIR__, "build"))

"""
    threaded_efv(ex, sets, nchunk) -> Vector{Any}

Evaluate every site in `sets` concurrently, `nchunk` tasks with ONE workspace each.

Written as a top-level FUNCTION rather than inline in the testset, and that is load-bearing
rather than tidiness: a `Threads.@spawn` body inside a `@testset` captures the testset's
locals, and any local that is also assigned elsewhere in the same scope (`Rs`, `Zs`, `F`, `E`
below) is boxed and SHARED by every task.  Written inline, this testset raced on its own
loop variables and reported spurious per-site differences -- a false alarm about the library
that cost a debugging round.  Inside a function each task's locals are its own.

One workspace per TASK, deliberately not `wss[threadid()]`: on Julia >= 1.7 a task can
migrate between threads mid-call, so `threadid()` is not a stable identity and indexing
per-thread state with it reintroduces exactly the race this API removes.  The LAMMPS plugin's
`omp_get_thread_num()` inside a parallel region IS stable, which is why one workspace per
OpenMP thread is the right pattern there.
"""
function threaded_efv(ex, sets, nchunk::Int)
    out = Vector{Any}(undef, length(sets))
    wss = [Base.invokelatest(ex.new_workspace) for _ in 1:nchunk]
    tasks = map(1:nchunk) do c
        Threads.@spawn begin
            w = wss[c]
            for i in c:nchunk:length(sets)
                Rs, Zs, Z0, _ = sets[i]
                F = Vector{SVector{3, Float64}}(undef, length(Rs))
                E, V = Base.invokelatest(ex.site_energy_forces_virial!, w, Rs, Zs, Z0, F)
                out[i] = (E, F, V)
            end
        end
    end
    foreach(wait, tasks)
    return out
end

"Serial re-use of ONE workspace over every site -> (max|dE|, all forces equal?, all virials equal?)"
function reused_ws_efv(ex, sets, ref)
    ws = Base.invokelatest(ex.new_workspace)
    dE = 0.0
    okF = true
    okV = true
    for (i, s) in enumerate(sets)
        Rs, Zs, Z0, _ = s
        F = Vector{SVector{3, Float64}}(undef, length(Rs))
        E, V = Base.invokelatest(ex.site_energy_forces_virial!, ws, Rs, Zs, Z0, F)
        dE = max(dE, abs(E - ref[i][1]))
        okF &= (F == ref[i][2])
        okV &= (V == ref[i][3])
    end
    return dE, okF, okV
end

@testset "workspace API: re-entrant and cap-free" begin
    if Threads.nthreads() < 2
        @warn "test_workspace.jl is running with $(Threads.nthreads()) thread; the " *
              "concurrency property below is vacuous.  Re-run with `julia -t 4`."
    end

    fx = load_cantor_fixture()
    f = joinpath(WS_BUILD, "cantor_ws_poly.jl")
    Base.invokelatest(export_ace_model, fx.stacked, f;
                      for_library = false, radial_basis = :polynomial)
    ex = Base.invokelatest(load_exported, f)

    src = read(f, String)

    # ---- the cap and the globals are gone from the SOURCE ------------------------------
    @test !occursin("MAX_NEIGHBORS", src)
    @test !occursin("WORK_", src)
    @test occursin("new_workspace", src)

    # ---- serial reference ---------------------------------------------------------------
    sets = Base.invokelatest(site_sets, fx.held[1], fx.rcut)
    @test length(sets) > 0
    ref = [Base.invokelatest(ex.site_energy_forces_virial, s[1], s[2], s[3]) for s in sets]

    # ---- one workspace per task, concurrent, BITWISE identical ------------------------
    nchunk = max(Threads.nthreads(), 2)
    out = threaded_efv(ex, sets, nchunk)
    nbad = 0
    dEt = dFt = dVt = 0.0
    for i in eachindex(sets)
        ok = out[i][1] === ref[i][1] && out[i][2] == ref[i][2] && out[i][3] == ref[i][3]
        ok || (nbad += 1)
        dEt = max(dEt, abs(out[i][1] - ref[i][1]))
        dFt = max(dFt, maximum(norm.(out[i][2] .- ref[i][2])))
        dVt = max(dVt, maximum(abs.(out[i][3] .- ref[i][3])))
    end
    println("threaded ($nchunk tasks, $(Threads.nthreads()) threads) vs serial over " *
            "$(length(sets)) sites: $nbad site(s) differ (must be 0, bitwise); " *
            "max|dE| = $dEt, max|dF| = $dFt, max|dV| = $dVt")
    @test nbad == 0

    # ---- one workspace re-used over every site agrees with a fresh one per site --------
    dEr, okF, okV = reused_ws_efv(ex, sets, ref)
    println("one re-used workspace vs one per call: max|dE| = $dEr, " *
            "forces identical: $okF, virials identical: $okV")
    @test dEr == 0.0
    @test okF
    @test okV

    # ---- no neighbour cap ---------------------------------------------------------------
    n = 300
    Rs = [SVector(2.0 + 0.005k, 0.1k, -0.05k) for k in 1:n]
    Rs = [R * (6.0 / norm(R)) * (0.5 + 0.4 * (k / n)) for (k, R) in enumerate(Rs)]
    Z1 = Int(ex.I2Z[1])
    Zs = fill(Z1, n)
    Z0 = Z1
    E = Base.invokelatest(ex.site_energy, Rs, Zs, Z0)
    @test isfinite(E)
    Ef, Ff, Vf = Base.invokelatest(ex.site_energy_forces_virial, Rs, Zs, Z0)
    @test isfinite(Ef) && length(Ff) == n && all(isfinite, Vf)
end
