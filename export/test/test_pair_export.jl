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
    @test occursin("pair_energy_d(r, iz0, jz)", src)   # actually called from the site loops

    # E0 + pair + many-body reproduced to 1e-12 against the full stack
    dE, dF, dV = Base.invokelatest(check_export, f, fx.stacked, fx.held, fx.rcut;
                                   tol = 1e-12, label = "task-1 :polynomial + pair vs full stack")
    @test dE <= 1e-12
    @test dF <= 1e-12
    @test dV <= 1e-12

    # and it must be a genuine improvement on the pair-less generator, which sat at 6.88 eV/Å
    @test dF < 1e-6
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
