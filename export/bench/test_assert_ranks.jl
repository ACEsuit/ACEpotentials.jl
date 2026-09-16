# test_assert_ranks.jl -- exercise gate_bench_libs.jl's `assert_ranks` on synthetic LAMMPS
# output, including the case it exists to catch.
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
# No LAMMPS, no MPI, no libraries: the function is lifted out of gate_bench_libs.jl by source
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
