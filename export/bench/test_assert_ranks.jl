# test_assert_ranks.jl -- exercise gate_bench_libs.jl's `assert_ranks` AND `output_is_finite`
# on synthetic LAMMPS output, including the cases they exist to catch and the cases on which
# earlier versions of them failed.
#
#   julia export/bench/test_assert_ranks.jl        # exit 0 = all cases behave as specified
#
# This file exists because of Task 4's own I5 finding: a gate nobody tested is a gate that can
# pass when the thing it checks did not happen.  `assert_ranks` is itself such a gate -- it is
# the only thing standing between gate C and a `mpirun` that launches N independent SERIAL
# jobs, each computing the whole cell and printing the same energy, which would make the
# 1-vs-2-rank comparison agree perfectly while proving nothing.  Adding that assertion without
# testing it would have repeated the defect it was added to remove.
#
# `output_is_finite` (gate L2's "and the energies stayed finite" clause) is here for the same
# reason and with a sharper history: its first version failed four HEALTHY libraries because a
# bare `inf` matches "Neighbor list **inf**o", and its second -- written to fix that -- dropped
# the very lines it was meant to read, so the clause could not fail at all and the gate printed
# "energies finite" having checked nothing.  Both of those are cases below.  A gate's FAILURE
# path is the half that has to be tested; on this task that lesson has now been paid for three
# times.
#
# No LAMMPS, no MPI, no libraries: the functions are lifted out of gate_bench_libs.jl by source
# and evaluated in a scratch module, so this cannot be fooled by a stale definition elsewhere.

const GATE_FILE = normpath(joinpath(@__DIR__, "gate_bench_libs.jl"))

src = read(GATE_FILE, String)
i = findfirst("function assert_ranks(", src)
i === nothing && error("assert_ranks not found in $GATE_FILE")
j = findfirst("\nend\n", src[i[1]:end])
j === nothing && error("could not find the end of assert_ranks in $GATE_FILE")
scratch = Module(:AssertRanksScratch)
Core.eval(scratch, Meta.parse(src[i[1] : i[1] + j[end] - 1]))
const AR = getfield(scratch, :assert_ranks)

# `output_is_finite` is a one-line definition, lifted the same way.
k = findfirst("output_is_finite(out::AbstractString) =", src)
k === nothing && error("output_is_finite not found in $GATE_FILE")
kend = findfirst("\n", src[k[1]:end])
knext = findfirst("\n", src[k[1] + kend[1] : end])
Core.eval(scratch, Meta.parse(src[k[1] : k[1] + kend[1] + knext[1] - 2]))
const FIN = getfield(scratch, :output_is_finite)

good2   = "1 by 1 by 2 MPI processor grid\nLoop time of 0.5 on 2 procs for 0 steps with 250 atoms\n"
serial  = "1 by 1 by 1 MPI processor grid\nLoop time of 0.5 on 1 procs for 0 steps with 250 atoms\n"
nogrid  = "Loop time of 0.5 on 2 procs for 0 steps with 250 atoms\n"
noloop  = "1 by 1 by 2 MPI processor grid\nERROR: something went wrong\n"
badgrid = "1 by 1 by 1 MPI processor grid\nLoop time of 0.5 on 2 procs for 0 steps with 250 atoms\n"

const CASES = [
    ("a real 2-rank run, expecting 2",                      good2,   2, :ok),
    ("a real 1-rank run, expecting 1",                      serial,  1, :ok),
    ("TWO INDEPENDENT SERIAL JOBS -- the hazard, expect 2",  serial,  2, :throw),
    ("no processor-grid line, expecting 2",                 nogrid,  2, :throw),
    ("no Loop-time line (the run died), expecting 2",        noloop,  2, :throw),
    ("2 procs but a 1x1x1 grid, expecting 2",               badgrid, 2, :throw),
]

# --- output_is_finite ---------------------------------------------------------------------
# The two historical failures are the first two cases; everything after them is a value that
# must (or must not) be seen.
const FIN_CASES = [
    ("v1 REGRESSION: LAMMPS' own \"Neighbor list info\" line",
     "Neighbor list info ...\n  update: every = 1 steps\n", true),
    ("v2 REGRESSION: a thermo block that HAS gone non-finite",
     "   Step  PotEng  TotEng\n        0  -6768.18  -6748.83\n" *
     "        20         nan          nan\n        40          inf   -3167.9414\n", false),
    ("a healthy thermo block",
     "   Step  PotEng  TotEng\n        0  -6768.18  -6748.83\n       100  -6804.51  -6748.85\n", true),
    ("an echoed mktempdir path containing the letters",
     "read_data /tmp/jl_3nan7/geom.data\nLoop time of 2.0 on 1 procs\n", true),
    ("the word infinity in prose",
     "cutoff is not infinity\n", true),
    ("nanosecond in prose",
     "timestep 1 nanosecond\n", true),
    ("a bare -inf value",
     "   Step  PotEng\n       20         -inf\n", false),
    ("a bare -nan value",
     "   Step  PotEng\n       20         -nan\n", false),
    ("NaN, mixed case",
     "   Step  PotEng\n       20          NaN\n", false),
    ("INF, upper case",
     "   Step  PotEng\n       20          INF\n", false),
]

let fails = 0
    for (name, out, want) in FIN_CASES
        got = FIN(out)
        ok = got == want
        ok || (fails += 1)
        println(ok ? "PASS " : "FAIL ", rpad(name, 54), " -> finite=", got, " (want ", want, ")")
    end
    fails == 0 || (println("OUTPUT_IS_FINITE_CHECK_FAILED ($fails case(s))"); exit(1))
    println("OUTPUT_IS_FINITE_CHECK_OK")
end

let fails = 0
    for (name, out, n, want) in CASES
        got = try
            (:ok, AR(out, n))
        catch e
            (:throw, sprint(showerror, e))
        end
        ok = got[1] == want
        ok || (fails += 1)
        detail = got[1] == :throw ?
            "  \"" * first(split(got[2], ". "))[1:min(end, 72)] * "...\"" : "  grid=$(got[2])"
        println(ok ? "PASS " : "FAIL ", rpad(name, 54), " -> ", got[1], detail)
    end
    println(fails == 0 ? "ASSERT_RANKS_CHECK_OK" : "ASSERT_RANKS_CHECK_FAILED ($fails case(s))")
    exit(fails == 0 ? 0 : 1)
end
