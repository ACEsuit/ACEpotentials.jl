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
# `site_energy` takes the value-only route (`compute_embeddings` -> `evaluate_Rnl`, `eval_ylm`)
# while the other two take the derivative route (`compute_embeddings_ed` -> `evaluate_Rnl_d`,
# `eval_ylm_ed`).  Those two routes do NOT agree bitwise, and this testset pins down exactly
# which half is responsible, because the two halves have very different status:
#
#   RADIAL  -- fixed in Task 2 and now gated BITWISE below (`d_Rnl == 0.0`).  Until then
#     `agnesi_transform` formed `s = (r - rin) / (req - rin)` while `agnesi_transform_d`
#     formed `s = (r - rin) * (1 / (req - rin))`; that 1-ulp difference, amplified by the
#     45-term recurrence, was the larger part of the site-energy spread (4.263256414560601e-14
#     eV/site).  Both now divide, exactly as `ET.eval_agnesi` does, and the radial embeddings
#     of the two routes are identical to the last bit.  Do not reintroduce a
#     reciprocal-multiply in one route only.
#
#   SPHERICAL HARMONICS -- NOT fixed, and not fixable inside export/.  `eval_ylm` is emitted
#     from `SpheriCart._codegen_Zlm` and `eval_ylm_ed` from `SpheriCart._codegen_Zlm_grads`
#     (codegen.jl:57,93): two independently generated expression trees for the same Zlm, which
#     agree only to roundoff.  Measured 3.552713678800501e-15 on the Ylm entries, and it is
#     the whole of the remaining 2.842170943040401e-14 eV/site energy spread.  That residual
#     is therefore recorded as `@test_broken` with an absolute 1e-12 gate, NOT asserted to
#     zero, and NOT papered over by relaxing anything to `≈`.
#
# Call this through `Base.invokelatest` (as check_export.jl does with `exported_efv`): the
# `ex.site_energy*` bindings do not exist in the world this file was compiled in.
function _site_function_spread(ex, fx)
    d_se = d_sef = d_F = d_Rnl = d_Ylm = 0.0
    nsites = 0
    for sys in fx.held
        for (Rs, Zs, Z0, _js) in site_sets(sys, fx.rcut)
            Ev, Fv, _Vv = ex.site_energy_forces_virial(Rs, Zs, Z0)
            Ef, Ff = ex.site_energy_forces(Rs, Zs, Z0)
            Es = ex.site_energy(Rs, Zs, Z0)
            d_se = max(d_se, abs(Es - Ev))
            d_sef = max(d_sef, abs(Ef - Ev))
            d_F = max(d_F, maximum(norm.(Ff .- Fv)))

            # Attribute the site_energy spread to the radial half or the Ylm half.  Both
            # entry points write the same WORK_* arrays, so the value-route results must be
            # COPIED before the derivative route overwrites them.
            Rnl_v, Ylm_v = ex.compute_embeddings(Rs, Zs, Z0)
            Rv, Yv = Array(Rnl_v), Array(Ylm_v)
            emb = ex.compute_embeddings_ed(Rs, Zs, Z0)
            d_Rnl = max(d_Rnl, maximum(abs.(Rv .- Array(emb[1]))))
            d_Ylm = max(d_Ylm, maximum(abs.(Yv .- Array(emb[3]))))
            nsites += 1
        end
    end
    return (; d_se, d_sef, d_F, d_Rnl, d_Ylm, nsites)
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
            "\n    max|site_energy_forces[2] - site_energy_forces_virial[2]| = $(s.d_F) / $(snp.d_F)" *
            "\n    max|Rnl(value route) - Rnl(derivative route)| = $(s.d_Rnl) / $(snp.d_Rnl)  [must be 0.0]" *
            "\n    max|Ylm(value route) - Ylm(derivative route)| = $(s.d_Ylm) / $(snp.d_Ylm)  [SpheriCart, not gated to 0]")
    flush(stdout)

    @test s.nsites == sum(length.(fx.held))

    # bitwise, and required to stay so
    @test s.d_sef == 0.0
    @test s.d_F == 0.0
    @test snp.d_sef == 0.0
    @test snp.d_F == 0.0

    # RADIAL: bitwise since Task 2 made agnesi_transform / agnesi_transform_d form `s`
    # identically (both by division, as ET.eval_agnesi does).  This is a hard gate.
    @test s.d_Rnl == 0.0
    @test snp.d_Rnl == 0.0

    # site_energy: still not bitwise, and the ONLY remaining cause is the spherical harmonics
    # (see the header).  Recorded rather than asserted away: it shows in the summary as Broken
    # and flips to a failure the day the two SpheriCart expression trees are unified.
    @test_broken s.d_se == 0.0
    @test s.d_se <= 1e-12
    @test snp.d_se <= 1e-12
    # ... and the attribution is asserted, not just asserted-about: the Ylm spread is nonzero
    # and of the same order as the site-energy spread, while the radial spread is exactly 0.
    @test s.d_Ylm > 0.0
    @test s.d_se <= 100 * s.d_Ylm
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
