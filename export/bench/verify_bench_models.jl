# verify_bench_models.jl -- export the benchmark reference models, GATE them at 1e-12, and
# print the measured deviation for each.  Nothing here is timed; this is the "verified before
# timed" half of the protocol (export/bench/README.md).  bench_parity.sh refuses to know
# anything about accuracy, so this script is what licenses a row.
#
#   cd <repo> && julia --project=export export/bench/verify_bench_models.jl [tags...]
#
# Tags (default: all four):
#   cantor_poly  cantor_h50  tial_poly  tial_h50
#
# Each tag writes bench_parity/<tag>_model.jl and compares it, at tol = 1e-12, against the
# reference its mode requires:
#   *_poly  -> fx.stacked                     = (ETOneBody, ETPairModel, ETACE)
#   *_h50   -> <model>_spline_stack(fx; 50)   = (ETOneBody, ETPairModel, splinified ETACE)
# and additionally REPORTS (never asserts) the Hermite error against the fitted stack, which is
# model error, not export error.

using Printf

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "bench_parity"); mkpath(OUT)

include(joinpath(REPO, "export", "test", "check_export.jl"))          # pulls in cantor_fixture
include(joinpath(REPO, "export", "test", "fixtures", "tial_fixture.jl"))
include(joinpath(REPO, "export", "src", "export_ace_model.jl"))

const TAGS = isempty(ARGS) ? ["cantor_poly", "cantor_h50", "tial_poly", "tial_h50"] : ARGS
const TOL = 1e-12

results = Dict{String, Any}()

function do_tag(tag)
    t0 = time()
    if startswith(tag, "cantor")
        fx = load_cantor_fixture()
        spline = Nspl -> cantor_spline_stack(fx; Nspl = Nspl)
    elseif startswith(tag, "tial")
        fx = load_tial_fixture()
        spline = Nspl -> tial_spline_stack(fx; Nspl = Nspl)
    else
        error("unknown tag $tag")
    end
    hermite = endswith(tag, "_h50")
    calc = hermite ? spline(50) : fx.stacked
    mode = hermite ? :hermite_spline : :polynomial
    file = joinpath(OUT, "$(tag)_model.jl")

    Base.invokelatest(export_ace_model, calc, file; for_library = true, radial_basis = mode)
    @printf("[%s] exported %s (%.1f MB, %.0f s)\n", tag, basename(file),
            filesize(file) / 2^20, time() - t0)
    flush(stdout)

    ref = hermite ? "E0 + pair + splinified(Nspl=50) ETACE" : "E0 + pair + ETACE (the fitted stack)"
    @printf("[%s] accuracy gate, reference = %s, tol = %.0e\n", tag, ref, TOL)
    dE, dF, dV = check_export(file, calc, fx.held, fx.rcut; tol = TOL, label = "$tag vs $ref")
    extra = nothing
    if hermite
        println("[$tag] REPORTED ONLY (model error of splinification, never asserted):")
        extra = check_export_report(file, fx.stacked, fx.held, fx.rcut;
                                    label = "$tag vs the FITTED stack")
    end
    results[tag] = (; file, dE, dF, dV, ref, extra, natoms = sum(length, fx.held))
    @printf("[%s] OK in %.0f s\n\n", tag, time() - t0); flush(stdout)
end

for tag in TAGS
    do_tag(tag)
end

println("=" ^ 100)
println("VERIFICATION SUMMARY (all gates at tol = $TOL; a row may be timed only if it PASSED)")
@printf("%-12s  %-42s  %11s  %11s  %11s\n", "tag", "reference", "max|dE|/at", "max|dF|", "max|dV|/at")
for tag in TAGS
    r = results[tag]
    @printf("%-12s  %-42s  %11.3e  %11.3e  %11.3e\n", tag, r.ref, r.dE, r.dF, r.dV)
end
for tag in TAGS
    r = results[tag]
    r.extra === nothing && continue
    @printf("%-12s  %-42s  %11.3e  %11.3e  %11.3e   (reported, not gated)\n",
            tag, "vs the FITTED stack", r.extra...)
end
println("DONE verify_bench_models.jl")
