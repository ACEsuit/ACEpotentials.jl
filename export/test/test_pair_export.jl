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
    # check_export_report measures without asserting, so the three @test lines below ARE the
    # gate; with check_export its internal @assert fires first and makes them unfailable.
    # (The pair-less generator sat at max|dF| = 6.878706238581598 eV/Å on this comparison.)
    dE, dF, dV = Base.invokelatest(check_export_report, f, fx.stacked, fx.held, fx.rcut;
                                   label = "task-1 :polynomial + pair vs full stack")
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
#   SPHERICAL HARMONICS -- still not fixable inside export/, but since Task 6 it no longer
#     reaches the energy.  `eval_ylm` is emitted from `SpheriCart._codegen_Zlm` and
#     `eval_ylm_ed` from `SpheriCart._codegen_Zlm_grads` (codegen.jl): two independently
#     generated expression trees for the same Zlm, which agree only to roundoff (measured
#     3.552713678800501e-15 on the Ylm entries, and `d_Ylm` below still measures it).  Until
#     B2 that was the whole of a 2.842170943040401e-14 eV/site energy spread between the two
#     routes, recorded as `@test_broken`.  B2's kernel accumulates `A` from the VALUE
#     embeddings on both routes -- the derivative route re-evaluates the edge only for the
#     forces -- so the energies are now bitwise equal and the spread is gated at 0.0.
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

            # Attribute the site_energy spread to the radial half or the Ylm half.  Since
            # Task 6 `compute_embeddings*` allocate their own output (they are diagnostics,
            # not the kernel: the kernel is per-neighbour and never materialises a full-width
            # Rnl), so no copy is needed before the derivative route runs.
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

    # site_energy is now BITWISE equal to site_energy_forces_virial's energy, and this is a
    # hard gate.  It was a `@test_broken` at 2.842170943040401e-14 until Task 6, with the
    # residual attributed to SpheriCart: `eval_ylm` and `eval_ylm_ed` are two independently
    # generated expression trees for the same Zlm, so the value route and the derivative
    # route saw Ylm values differing by ~3.6e-15 and the energies inherited it.
    #
    # B2 removed the cause rather than the symptom.  The per-neighbour kernel accumulates `A`
    # in ONE pass (`_embed_val!`, values only) on BOTH routes -- the derivative route
    # re-evaluates the edge in pass 2 for the forces, and the energy never sees `eval_ylm_ed`
    # at all.  So the two routes now contract identical `A`, `AA` and `B`.
    #
    # The Ylm discrepancy itself has NOT gone away: `d_Ylm` below is still nonzero, and it is
    # checked to be so, precisely so that this testset keeps measuring the thing it names
    # instead of quietly passing because the diagnostic stopped working.
    @test s.d_se == 0.0
    @test snp.d_se == 0.0

    # ATTRIBUTION -- the load-bearing part.  The pair-less export of the SAME model, measured
    # in the same sweep, must agree exactly: if a future change made the pair term contribute
    # asymmetrically to the two routes, `s` would move away from `snp` and this would fail.
    @test s.d_se == snp.d_se

    # The value/derivative Ylm routes still disagree (SpheriCart, upstream) -- so the equality
    # above is a property of the KERNEL, not an artefact of the diagnostic having gone quiet.
    @test s.d_Ylm > 0.0
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
