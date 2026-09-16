# verify_bench_models.jl -- export the benchmark reference models, GATE the GENERATED SOURCE
# at 1e-12, and print the measured deviation for each.  Nothing here is timed.
#
# THIS IS ONLY THE FIRST OF THREE LINKS.  The plan's global constraints gate three things,
# and this script covers exactly one of them:
#
#   1. generated Julia code vs the ETACE/Stacked calculator      1e-12   <- HERE
#   2. the compiled library through the Python C API vs Julia    1e-12   <- gate_bench_libs.jl
#   3. `pair_style ace` in LAMMPS vs Julia, 1 and 2 MPI ranks    1e-10 / 1e-12
#                                                                        <- gate_bench_libs.jl
#
# A library that has passed (1) alone has NOT been shown to compute what the source says: a
# juliac or cpu_target miscompilation would change what it computes, and therefore what it
# costs, without touching the generated `.jl` that (1) checks.  Run gate_bench_libs.jl too.
#
# Output per tag: `bench_parity/<tag>_model.jl` and the gate manifest
# `bench_parity/<tag>.gated`, which bench_parity.sh requires before it will time anything.
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

using Printf, SHA, Dates

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

    # The gate MANIFEST.  bench_parity.sh refuses to time a library whose manifest is
    # missing or whose sha256 does not match, so a gate that was never run, or a library
    # rebuilt after the gate, cannot silently produce a timing row.  This file is rewritten
    # from scratch here (the source gate is the first link in the chain);
    # gate_bench_libs.jl APPENDS the library-level gates and `lib_sha256` to it.
    open(joinpath(OUT, "$(tag).gated"), "w") do io
        println(io, "# gate manifest for '$tag' -- written by export/bench/verify_bench_models.jl")
        println(io, "# consumed by export/bench/bench_parity.sh; do not hand-edit")
        println(io, "tag=$tag")
        println(io, "date=", Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"))
        println(io, "model_file=$file")
        println(io, "model_sha256=", bytes2hex(open(sha256, file)))
        println(io, "source_gate_reference=$ref")
        println(io, "source_gate_tol=$TOL")
        @printf(io, "source_gate_dE_per_atom=%.6e\n", dE)
        @printf(io, "source_gate_dF=%.6e\n", dF)
        @printf(io, "source_gate_dV_per_atom=%.6e\n", dV)
        println(io, "source_gate=PASS")
        if extra !== nothing
            @printf(io, "# reported only, never asserted: vs the FITTED stack dE/atom=%.3e dF=%.3e dV/atom=%.3e\n",
                    extra...)
        end
    end
    @printf("[%s] wrote %s\n", tag, joinpath(OUT, "$(tag).gated"))
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
