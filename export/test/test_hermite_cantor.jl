# Task 2: `:hermite_spline` export of the fitted Cantor model, at two knot counts.
#
# REFERENCES -- there are two, and confusing them is the mistake this file exists to prevent:
#
#   * GATED (1e-12): the SPLINIFIED stack `cantor_spline_stack(fx; Nspl)`.  That stack is
#     `(ETOneBody, ETPairModel, splinified ETACE)` -- the pair term is NOT splinified and the
#     generator emits it in both radial modes, so a pair-less reference would re-measure the
#     6.9 eV/Å defect Task 1 removed and blame it on Hermite interpolation.
#
#   * REPORTED, NEVER GATED: the FITTED stack `fx.stacked`.  The difference there is
#     splinification error of the MODEL, not error of the EXPORT.  `verify_cantor/log.chain`
#     records it as ~2.3e-4 eV/Å at Nspl = 50 and ~3.0e-6 eV/Å at Nspl = 200.  Asserting a
#     Hermite export against the fitted model would be asserting a model approximation, and
#     the only way to make such an assertion pass is to loosen a tolerance.  Do not.
#
# The 43 pre-existing Julia tests only ever exported single-species, pair-less models, so
# before this file nothing checked a Hermite export of a real multi-species fit with a live
# pair term at all.

using Test
using StaticArrays

include(joinpath(@__DIR__, "check_export.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))

@testset "Hermite export of the fitted Cantor model (NZ=5)" begin
    fx = load_cantor_fixture()
    build = mkpath(joinpath(@__DIR__, "build"))

    for Nspl in (50, 200)
        spl = cantor_spline_stack(fx; Nspl = Nspl)
        @test length(spl.calcs) == 3            # the pair term must be in the reference
        @test any(occursin("Pair", string(nameof(typeof(c.model)))) for c in spl.calcs)

        f = joinpath(build, "cantor_hermite$(Nspl).jl")
        Base.invokelatest(export_ace_model, spl, f; radial_basis = :hermite_spline)
        src = read(f, String)
        @test occursin("HERMITE CUBIC SPLINE RADIAL BASIS", src)
        @test occursin("const PAIR_C", src)                 # pair term emitted in this mode too
        @test occursin("zz2pair_sym", src) == false         # one pair-index convention only

        dE, dF, dV = Base.invokelatest(check_export, f, spl, fx.held, fx.rcut;
                                       tol = 1e-12,
                                       label = "Cantor :hermite_spline(Nspl=$Nspl) vs SPLINIFIED stack")
        @test dE <= 1e-12
        @test dF <= 1e-12
        @test dV <= 1e-12

        # Informational ONLY -- model error, never gated.  See the header.
        fit = Base.invokelatest(check_export_report, f, fx.stacked, fx.held, fx.rcut;
                                label = "Cantor :hermite_spline(Nspl=$Nspl) vs FITTED stack [informational, log.chain: 2.3e-4 @50, 3.0e-6 @200]")
        @info "Cantor Hermite error against the FITTED model -- reported, never gated" Nspl dE_atom = fit[1] dF = fit[2] dV_atom = fit[3]
    end
end
