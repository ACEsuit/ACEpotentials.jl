# Task 1: the exported model must contain the ETPairModel term.
#
# Before this test existed the generator silently dropped the pair calculator from a
# StackedCalculator((ETOneBody, ETPairModel, ETACE)), so every exported library was
# one-body + many-body only.  Measured against the full Cantor stack that error was
# max|dF| = 6.878706238581598 eV/Å -- larger than the held-out forces themselves.
#
# Reference used here: `fx.stacked`, i.e. the FULL (ETOneBody, ETPairModel, ETACE) stack.
# `cantor_mb_stack(fx)` (E0 + many-body only) is deliberately NOT the reference any more.
#
# check_export's three maxima are normalised differently (dE and dV per atom, dF absolute);
# see the docstrings in check_export.jl.  The tolerance is 1e-12 and must never be loosened.

using Test
using StaticArrays

include(joinpath(@__DIR__, "check_export.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))

# Sentinel for the refusal test.  It has to live at the top level of this file: a `@testset`
# body is a local scope and `struct` is not allowed there.  (`include` always evaluates at
# module top level, so this is legal even though runtests.jl includes this file from inside
# a function.)
struct NotAnExportableModel end

@testset "Pair potential is exported" begin
    fx = load_cantor_fixture()
    build = mkpath(joinpath(@__DIR__, "build"))
    f = joinpath(build, "cantor_pair_poly.jl")

    Base.invokelatest(export_ace_model, fx.stacked, f;
                      for_library = false, radial_basis = :polynomial)
    src = read(f, String)

    # the pair constants and kernels are present
    @test occursin("const N_PAIRPOLYS", src)
    @test occursin("const PAIRPOLY_A", src)
    @test occursin("const PAIRPOLY_B", src)
    @test occursin("const PAIRPOLY_C", src)
    @test occursin("const PAIR_ENV_RCUT", src)
    @test occursin("const PAIR_ENV_P", src)
    @test occursin("const PAIR_C", src)
    @test occursin("const PAIR_TRANSFORM_PARAMS", src)
    @test occursin("function pair_energy_d", src)

    # E0 + pair + many-body reproduced to 1e-12 against the full stack
    # (the pair-less generator sat at max|dF| = 6.878706238581598 eV/Å here)
    dE, dF, dV = Base.invokelatest(check_export, f, fx.stacked, fx.held, fx.rcut;
                                   tol = 1e-12, label = "task-1 :polynomial + pair vs full stack")
    @test dE <= 1e-12
    @test dF <= 1e-12
    @test dV <= 1e-12
end

# check_export only ever calls `site_energy_forces_virial`, so without this testset the pair
# term emitted into `site_energy` and into `site_energy_forces` would be exercised in this
# task only through the zero stubs -- yet both are @ccallable C entry points
# (write_c_interface.jl) that the Python and LAMMPS paths call.  Transposing the species
# arguments in one of them, or dropping its `Ei += ep`, would leave the rest of this suite
# green at 1e-12 and only surface in Task 4's library comparison.
#
# `site_energy_forces` and `site_energy_forces_virial` share their whole evaluation route, so
# they are compared BITWISE and must stay that way -- do not relax those to `≈`.
#
# `site_energy` takes the value-only route (`compute_embeddings` -> `evaluate_Rnl`) while the
# other two take the derivative route (`compute_embeddings_ed` -> `evaluate_Rnl_d`), and those
# two form the transform variable differently: `agnesi_transform` computes
# `s = (r - rin) / (req - rin)` whereas `agnesi_transform_d` computes
# `s = (r - rin) * (1 / (req - rin))`, which is a 1-ulp difference that the 45-term recurrence
# amplifies.  That is PRE-EXISTING many-body behaviour, not the pair term: the run below
# measures the same divergence on a pair-less export of the same model.  It is recorded as
# `@test_broken` (visible in the summary, and it flips to a failure the day someone makes the
# two agree) and gated absolutely at 1e-12 per site, which is ~20x the measured divergence and
# orders of magnitude below either failure mode this testset exists to catch.
#
# Call this through `Base.invokelatest` (as check_export.jl does with `exported_efv`): the
# `ex.site_energy*` bindings do not exist in the world this file was compiled in.
function _site_function_spread(ex, fx)
    d_se = d_sef = d_F = 0.0
    nsites = 0
    for sys in fx.held
        for (Rs, Zs, Z0, _js) in site_sets(sys, fx.rcut)
            Ev, Fv, _Vv = ex.site_energy_forces_virial(Rs, Zs, Z0)
            Ef, Ff = ex.site_energy_forces(Rs, Zs, Z0)
            Es = ex.site_energy(Rs, Zs, Z0)
            d_se = max(d_se, abs(Es - Ev))
            d_sef = max(d_sef, abs(Ef - Ev))
            d_F = max(d_F, maximum(norm.(Ff .- Fv)))
            nsites += 1
        end
    end
    return (; d_se, d_sef, d_F, nsites)
end

@testset "site_energy / _forces / _forces_virial agree, with a live pair term" begin
    fx = load_cantor_fixture()
    f = joinpath(@__DIR__, "build", "cantor_pair_poly.jl")
    @test isfile(f)   # produced by the testset above
    s = Base.invokelatest(_site_function_spread, Base.invokelatest(load_exported, f), fx)

    # same sweep on a pair-less export of the same model, to attribute the site_energy spread
    fnp = joinpath(@__DIR__, "build", "cantor_nopair_poly.jl")
    Base.invokelatest(export_ace_model, cantor_mb_stack(fx), fnp;
                      for_library = false, radial_basis = :polynomial)
    snp = Base.invokelatest(_site_function_spread, Base.invokelatest(load_exported, fnp), fx)

    println("site-function cross-check over $(s.nsites) sites " *
            "(pair export / pair-less export of the same model):" *
            "\n    max|site_energy        - site_energy_forces_virial[1]| = $(s.d_se) / $(snp.d_se)" *
            "\n    max|site_energy_forces[1] - site_energy_forces_virial[1]| = $(s.d_sef) / $(snp.d_sef)" *
            "\n    max|site_energy_forces[2] - site_energy_forces_virial[2]| = $(s.d_F) / $(snp.d_F)")
    flush(stdout)

    @test s.nsites == sum(length.(fx.held))

    # bitwise, and required to stay so
    @test s.d_sef == 0.0
    @test s.d_F == 0.0
    @test snp.d_sef == 0.0
    @test snp.d_F == 0.0

    # site_energy: known pre-existing spread, gated absolutely
    @test_broken s.d_se == 0.0
    @test s.d_se <= 1e-12
    # and it is not the pair term's doing: the pair-less export shows the same magnitude
    @test snp.d_se > 0.0
    @test s.d_se <= 10 * max(snp.d_se, eps())
end

@testset "Unknown calculator in the stack is refused" begin
    fake = (; model = NotAnExportableModel(), ps = nothing, st = nothing)
    @test_throws ErrorException Base.invokelatest(
        export_ace_model, ETM.StackedCalculator((fake,)), tempname())

    # a stack with no ETACE at all is still refused
    fx = load_cantor_fixture()
    onebody, pair, _ace = cantor_substacks(fx)
    @test_throws ErrorException Base.invokelatest(
        export_ace_model, ETM.StackedCalculator((onebody, pair)), tempname())
end
