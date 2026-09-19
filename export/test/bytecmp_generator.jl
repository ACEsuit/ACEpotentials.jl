# bytecmp_generator.jl -- is the SHIPPED DEFAULT's generated SOURCE byte-identical to the
# reference commit's?
#
#     EXPORT_REF_SHA=b826c831 taskset -c 0-30 julia --project=export \
#         export/test/bytecmp_generator.jl 2>&1 | tee bench_parity/bytecmp_task8.log
#
# WHY THIS EXISTS, AND WHY IT IS A COMMITTED FILE.
#
# The parity gate (`test_generator_parity.jl`) compares EVALUATED NUMBERS: it exports the same
# calculator through both generators, runs both, and asserts the energies, forces and virials
# agree.  This script makes the much stronger comparison -- that the two generators emit the
# same BYTES -- and prints `EXPORT_BUILD_ID` from each.  That matters for two claims the plan
# leans on and neither the parity gate nor a compiled-library check can make:
#
#   1. "Nothing needs re-timing or re-compiling."  If the source is identical then the standing
#      rows measure the current generator, not a predecessor -- no argument about whether a
#      whitespace or comment change could have moved the optimiser is needed.
#   2. "The build stamp is still valid."  `EXPORT_BUILD_ID` is derived from the emitted body and
#      is what the compiled library's build stamp gates, so an unchanged id means a regenerated
#      model file still matches a library compiled before the change.
#
# Task 7 made both claims from a script that lived only in a scratch directory -- the same
# artefact gap its own review had raised about two README tables, and the gap this file closes.
# It is worth being blunt about what that gap cost when it was closed elsewhere in this plan:
# committing the profile script revealed that a published table had been wrong by ~17 %.  A
# claim whose evidence is not reproducible is a claim nobody can check, including its author.
#
# WHAT RUNNING IT ALSO RUNS.  Including `test_generator_parity.jl` executes that file's
# top-level `@testset` (~3.5 min, 39 assertions) before the byte comparison begins.  That is
# deliberate and not worth engineering around: the two statements belong together, and a byte
# comparison that passed while the parity gate failed would mean the harness, not the
# generator, was broken.  `EXPORT_REF_SHA` is mandatory there and therefore mandatory here.
#
# WHAT IT COVERS.  The two `:polynomial` benchmark models, `for_library = true` (the form the
# build stamp gates).  It does NOT cover `aa_products = :dag`, which is off by default and
# whose emitted source is expected to differ from any commit before Task 8's dedent fix --
# see the finding.
#
# IT USED TO COVER FOUR.  `cantor_h50` and `tial_h50` were the `:hermite_spline` exports of
# the same two models; that mode was removed (export/bench/FINDINGS_parity.md §7), so those
# models cannot be exported at all any more and the cases are GONE rather than skipped.
# Their absence is stated in the run's own output, below, so that "two cases" is never read
# as "two cases silently failed to build".  A reference commit from before the removal still
# emits them -- `export_with` checks the old generator out of git -- which is exactly why
# leaving them in would have compared a mode that ships against one that does not.
using Printf

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "check_export.jl"))
include(joinpath(@__DIR__, "fixtures", "tial_fixture.jl"))
include(joinpath(@__DIR__, "test_generator_parity.jl"))   # export_with / export_current / REF_SHA

const OUTDIR = mkpath(joinpath(@__DIR__, "build", "bytecmp"))

const CASES = [
    ("cantor_poly", () -> load_cantor_fixture().stacked, :polynomial),
    ("tial_poly",   () -> load_tial_fixture().stacked, :polynomial),
]

"The cases this script used to run, and why it does not: printed, never silently absent."
const RETIRED_CASES = ["cantor_h50", "tial_h50"]

function bytecmp_case(nm, mk, mode)
    calc = mk()
    fo = joinpath(OUTDIR, "$(nm)_ref.jl")
    fn = joinpath(OUTDIR, "$(nm)_new.jl")
    export_with(REF_SHA, calc, fo; mode = mode, for_library = true)
    export_current(calc, fn; mode = mode, for_library = true)
    a = read(fo); b = read(fn)
    same = a == b
    @printf("%-12s ref %8d B   new %8d B   BYTE-IDENTICAL: %s\n", nm, length(a), length(b), same)
    if !same
        # Print the first few differing lines rather than only the verdict: when this last
        # fired, the difference was not the one being looked for (an indentation change) but a
        # 5-line comment the restructuring had dropped, and only the diff showed that.
        la = split(String(copy(a)), '\n'); lb = split(String(copy(b)), '\n')
        n = 0
        for i in 1:min(length(la), length(lb))
            if la[i] != lb[i]
                n += 1
                n <= 5 && println("   line $i:\n     ref: ", repr(la[i]), "\n     new: ", repr(lb[i]))
            end
        end
        println("   differing lines: ", n, "  (ref ", length(la), " lines, new ", length(lb), ")")
    end
    idof(x) = match(r"const EXPORT_BUILD_ID = (0x[0-9a-f]+)", String(copy(x))).captures[1]
    ida, idb = idof(a), idof(b)
    @printf("             EXPORT_BUILD_ID ref %s  new %s  %s\n",
            ida, idb, ida == idb ? "(unchanged)" : "*** CHANGED ***")
    return same && ida == idb
end

println("\nbytecmp_generator.jl -- shipped default vs $REF_SHA, for_library = true\n")
println("cases: ", join(first.(CASES), ", "))
println("retired with the :hermite_spline mode, NOT skipped and NOT failed: ",
        join(RETIRED_CASES, ", "), "\n")
ok = true
for (nm, mk, mode) in CASES
    global ok &= bytecmp_case(nm, mk, mode)
end
println()
if ok
    println("ALL $(length(CASES)) SHIPPED MODELS BYTE-IDENTICAL, EXPORT_BUILD_ID UNCHANGED vs $REF_SHA.")
else
    println("NOT byte-identical -- see the per-case diff above. Nothing is wrong with the")
    println("generator merely because of this; but any standing timing row or build stamp")
    println("taken before the change must be re-justified rather than assumed to carry over.")
end
exit(ok ? 0 : 1)
