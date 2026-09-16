# gate_bench_libs.jl -- the LIBRARY-level gates the global constraints require, run on the
# four compiled benchmark libraries themselves.
#
#   cd <repo> && julia --project=export export/bench/gate_bench_libs.jl [tags...]
#
# verify_bench_models.jl gates the GENERATED SOURCE at 1e-12.  That says nothing about the
# `.so` that bench_parity.sh actually times: a juliac or `cpu_target` miscompilation changes
# what the library computes -- and therefore what it costs -- while leaving the generated
# `.jl` untouched.  Tasks 5-7 compare generated Julia files to each other, so such a fault
# would survive all the way to Task 8's final table.  This script closes that:
#
#   A. `pair_style ace` in LAMMPS  vs the exported model evaluated in Julia   tol 1e-10
#   B. the library through the Python C API vs the same Julia numbers         tol 1e-12
#   C. two MPI ranks vs one rank, same library, same geometry
#        energy |dE|/|E| tol 1e-13 (relative), forces max|dF| tol 1e-12 (absolute)
#
# All three run on the SAME geometry, and on the benchmark's OWN box: LAMMPS builds the box
# from `box_<model>.lmp` with `-var cells 5` (same lattice, same species fractions, same
# neighbour count per atom, a cell still wider than 2*rcut) and writes it out with
# `write_data` in a separate serial step; all three gates then `read_data` that ONE file, so
# the atom ids are identical across rank counts and "identical coordinates" is checked rather
# than assumed.  Using the timing box at full size would make the interpreted Julia
# reference take minutes per configuration for no extra coverage.
#
# Results are APPENDED to `bench_parity/<tag>.gated`, together with the library's sha256,
# which is what bench_parity.sh's interlock checks.

using Printf, SHA, Dates, LinearAlgebra, StaticArrays, Unitful
using Unitful: ustrip, @u_str
using AtomsBase: position

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const BENCH = @__DIR__
const OUT = joinpath(REPO, "bench_parity")

include(joinpath(REPO, "export", "test", "lammps_harness.jl"))
include(joinpath(REPO, "export", "test", "check_export.jl"))

const TAGS = isempty(ARGS) ? ["cantor_poly", "cantor_h50", "tial_poly", "tial_h50"] : ARGS
const CELLS = parse(Int, get(ENV, "ACE_GATE_CELLS", "5"))
const TOL_LAMMPS = 1e-10
const TOL_LIB = 1e-12
# ---------------------------------------------------------------------------------------
#  GATE C's METRICS -- READ THIS BEFORE CHANGING EITHER TOLERANCE
#
#  | quantity | definition                  | unit   | gate      | why                     |
#  |----------|-----------------------------|--------|-----------|-------------------------|
#  | energy   | |E_2ranks - E_1rank| / |E|  | -      | 1e-13 REL | extensive; see below    |
#  | forces   | max_i ||F_i - Fref_i||      | eV/A   | 1e-12 ABS | intensive; see below    |
#
#  The energy is compared RELATIVELY and the force ABSOLUTELY -- the same split
#  export/test/check_export.jl draws, and for the same reason.  A force is intensive: dividing
#  it by anything would weaken the gate on a larger cell.  A total energy is extensive AND, for
#  an ACE model, dominated by the one-body reference energies, so its magnitude is a property
#  of the chemistry rather than of the code under test.
#
#  WHY NOT AN ABSOLUTE PER-ATOM ENERGY GATE (what this script did first, and what
#  export/lammps/test/compare_dump.py still does).  E0(Ti) = -1586.02 eV/atom against
#  E0(Cr) ~ -14.4 makes a 250-atom TiAl cell total -216027 eV and a 500-atom Cantor cell
#  -6768 eV.  A flat 1e-12 eV/atom gate is then 8.6 ulp of the total on TiAl and 550 ulp on
#  Cantor -- two orders of magnitude tighter, in relative terms, on one model than the other,
#  purely because of the reference energies.  TiAl measured 34 ulp (9.90e-10 eV, 3.96e-12
#  eV/atom) and therefore "failed" a gate that no correct implementation could meet: summing
#  ~2000 terms in a different order drifts O(sqrt N) ~ 45 ulp.  That is a defective gate, not a
#  defective library -- the forces from the same runs agreed to 7.6e-14, and 2 ranks and 4
#  ranks produced the IDENTICAL total, i.e. a serial-vs-parallel summation path, not
#  accumulation across domains.
#
#  NOT DONE, DELIBERATELY: subtracting sum(E0) before comparing.  That removes a term from the
#  comparison in order to make it pass, which is the weakening the plan's constraints forbid.
#  Keeping E0 in and changing the metric's DIMENSION is the honest fix.
#
#  MEASURED HEADROOM under the relative gate (2026-09-16, this host):
#      tial_poly    4.581e-15   (22x inside 1e-13)     tial_h50    0.0
#      cantor_poly  1.478e-15   (68x inside 1e-13)     cantor_h50  0.0
#  A real 1-vs-2-rank discrepancy -- a ghost atom missing from one domain -- is an eV-scale
#  effect, i.e. 1e-5 relative here, ten orders of magnitude above the gate.
#
#  RESIDUAL RISK, stated so it is not discovered later: an absolute energy error below
#  1e-13*|E| passes.  That is 2.2e-8 eV total (8.6e-11 eV/atom) for the 250-atom TiAl cell and
#  6.8e-10 eV total (1.4e-12 eV/atom) for the 500-atom Cantor cell.  Nothing else re-checks the
#  rank-to-rank energy, so this gate is the only thing standing behind it.  All three figures
#  -- relative, absolute total, and per atom -- are printed on every call and recorded in the
#  manifest, with the gated one labelled, so a later reader can apply any other criterion.
const TOL_MPI_E_REL = 1e-13      # gated: |dE| / |E_total|
const TOL_MPI_F = 1e-12          # gated: max|dF|, absolute

const SPEC = Dict(
    "cantor" => (box = "box_cantor.lmp", elements = (:Cr, :Mn, :Fe, :Co, :Ni),
                 species = "Cr Mn Fe Co Ni", typemap = "1:24,2:25,3:26,4:27,5:28"),
    "tial"   => (box = "box_tial.lmp", elements = (:Ti, :Al),
                 species = "Ti Al", typemap = "1:22,2:13"))
"""
    model_of(tag) -> "cantor" | "tial"

Which reference model a tag belongs to.  ERRORS on anything unrecognised rather than falling
through to a default: the `SPEC` entry it selects carries the box file, the element tuple, the
`pair_coeff` species list and the LAMMPS-type-to-Z map, so a silent default would gate a new
library against the wrong geometry, the wrong species and the wrong Z map, and every number it
printed would look perfectly healthy.  `bench_parity.sh` (BOX inference) and
`verify_bench_models.jl` (fixture selection) both already error on exactly this condition; this
was the one of the three that did not.

This is not hypothetical.  README records that neither benchmark model exercises a dense
`RBASIS_W_k` (both are `init_Wradial = :onehot`), so Task 5 has to introduce a tag of its own --
and it would have hit the silent fallthrough.
"""
function model_of(tag)
    startswith(tag, "cantor") && return "cantor"
    startswith(tag, "tial") && return "tial"
    error("""unknown tag '$tag': cannot tell which reference model it belongs to.
          Tags must start with "cantor" or "tial" so that the box file, element tuple,
          pair_coeff species list and LAMMPS-type -> Z map can be selected.
          To add a model, add a SPEC entry here and a prefix to this function --
          do NOT let a new tag fall through to an existing model's geometry.""")
end

const PLUGIN = get(ENV, "PLUGIN", joinpath(REPO, "verify_cantor", "plugin_build", "aceplugin.so"))
const MPIRUN = let c = get(ENV, "ACE_MPIRUN", "")
    !isempty(c) ? c :
    something(Sys.which("mpirun"),
              "/software/easybuild/software/OpenMPI/4.1.6-GCC-13.2.0/bin/mpirun")
end

"Run a LAMMPS input under `env`, returning the combined output (never throwing)."
function run_lmp(exe, env, input::AbstractString, file::AbstractString; ranks::Int = 1)
    write(file, input)
    buf = IOBuffer()
    cmd = ranks == 1 ? `$exe -in $file` : `$MPIRUN -np $ranks $exe -in $file`
    ok = try
        success(pipeline(setenv(cmd, env); stdout = buf, stderr = buf))
    catch e
        @error "LAMMPS failed to start" file exception = e
        false
    end
    out = String(take!(buf))
    ok || (out *= "\nLAMMPS_EXIT_NONZERO\n")
    return out
end

parse_ace_energy(out) = (m = match(r"ACE_ENERGY\s+(\S+)", out); m === nothing ? nothing : parse(Float64, m.captures[1]))

"""
    assert_ranks(out, n) -> String

Assert, from LAMMPS' OWN output, that the run really used `n` MPI ranks and really split the
cell `n` ways.  THROWS on a mismatch -- it must never skip.

Gate C is the only rank-sensitive check in this harness, and without this it is a gate that
passes when the thing it tests did not happen: a `mpirun` belonging to a different MPI than the
one LAMMPS was linked against launches `n` INDEPENDENT SERIAL jobs, each of which computes the
whole cell, writes the same dump and prints the same energy.  The 1-vs-2-rank comparison then
agrees perfectly while proving nothing about ghost atoms or the half/full neighbour list.  That
is not hypothetical on this host: `export/test/test_mpi.jl` (Task 3) found the entire MPI group
had been skipping silently, and `export/lammps/test/run_two_ranks.sh` guards the same way.

Two independent facts are checked:
  * `Loop time of <t> on <n> procs` -- the rank count LAMMPS itself reports (the pattern
    test_mpi.jl:89 uses), which is `1` for each of `n` independent serial jobs;
  * `<a> by <b> by <c> MPI processor grid` with a*b*c == n -- the decomposition actually built,
    which catches a run that has n ranks but was not decomposed (the check run_two_ranks.sh
    makes).

Returns the processor-grid string, for the manifest, so Task 8 can see it was checked.
"""
function assert_ranks(out::AbstractString, n::Int)
    m = match(r"Loop time of \S+ on (\d+) procs", out)
    m === nothing && error("no `Loop time of ... on N procs` line in the $(n)-rank output -- " *
                           "the run did not complete, so the rank count cannot be confirmed")
    got = parse(Int, m.captures[1])
    got == n || error("LAMMPS reports $got MPI rank(s) but $n were requested. Either mpirun " *
                      "belongs to a different MPI than LAMMPS was linked against (in which " *
                      "case it launched $n independent SERIAL jobs and gate C proves nothing), " *
                      "or the launch failed. Set ACE_MPIRUN to the mpirun matching " *
                      "the LAMMPS build.")
    g = match(r"(\d+) by (\d+) by (\d+) MPI processor grid", out)
    g === nothing && error("no `N by N by N MPI processor grid` line in the $(n)-rank output")
    grid = parse.(Int, g.captures)
    prod(grid) == n || error("LAMMPS built a $(join(grid, "x")) processor grid = $(prod(grid)) " *
                             "domains for $n requested ranks; the cell was not split $n ways, " *
                             "so gate C would not exercise the ghost-atom path")
    return join(grid, "x")
end

results = Dict{String, Any}()

for tag in TAGS
    mdl = model_of(tag)
    sp = SPEC[mdl]
    lib = joinpath(OUT, "libace_$(tag).so")
    model_file = joinpath(OUT, "$(tag)_model.jl")
    manifest = joinpath(OUT, "$(tag).gated")
    for f in (lib, model_file, manifest)
        isfile(f) || error("$f missing -- run verify_bench_models.jl $tag and compile_bench_libs.sh $tag first")
    end
    work = mkpath(joinpath(OUT, "gate_$(tag)"))
    geom = joinpath(work, "geom.data")
    dump1 = joinpath(work, "forces1.dump")
    dump2 = joinpath(work, "forces2.dump")

    env = ace_runtime_env(dirname(lib))
    exe, env = find_lammps_exe(env; extra_candidates =
        [joinpath(homedir(), "lammps", "lammps-22Jul2025", "build", "lmp")])
    isempty(exe) && error("no working LAMMPS executable found (see the rejection lines above)")

    # Step 0: build and perturb the cell ONCE, serially, and write it out.  Both timed runs
    # then `read_data` that file.  This is not tidiness: `create_atoms` assigns atom IDs
    # per-domain, so the SAME physical configuration gets a different ID -> position mapping
    # under two ranks than under one, and a dump compared by atom id then shows forces
    # differing by whole eV/A while the total energy stays bit-identical.  (Measured: 5.46
    # eV/A on Cantor, 17.75 on TiAl, dE/atom exactly 0.)  `export/lammps/test/run_two_ranks.sh`
    # makes the same point about `displace_atoms ... random`.
    build_input = """
    variable cells index $CELLS
    include $(joinpath(BENCH, sp.box))
    write_data $geom
    """
    out0 = run_lmp(exe, env, build_input, joinpath(work, "build.lmp"))
    (occursin("ERROR", out0) || !isfile(geom)) && error("[$tag] geometry build failed:\n$out0")

    lmp_input(dump) = """
    units metal
    atom_style atomic
    boundary p p p
    read_data $geom
    plugin load $PLUGIN
    pair_style ace
    pair_coeff * * $lib $(sp.species)
    variable e equal pe
    dump d all custom 1 $dump id type x y z fx fy fz
    dump_modify d sort id format float %.17g
    run 0
    print "ACE_ENERGY \$(v_e:%.17g)"
    """

    # ---------------- A. LAMMPS (1 rank) vs the exported model in Julia, 1e-10 ------------
    out1 = run_lmp(exe, env, lmp_input(dump1), joinpath(work, "gate1.lmp"))
    occursin("ERROR", out1) && error("[$tag] LAMMPS 1-rank gate run failed:\n$out1")
    E_lmp = parse_ace_energy(out1)
    d1 = read_lammps_dump(dump1)
    sys = read_lammps_data(geom, sp.elements)
    N = length(sys)

    # identical coordinates, checked not assumed
    Xdata = [SVector{3,Float64}(ustrip.(u"Å", p)) for p in position(sys, :)]
    maxdx = maximum(maximum(abs.(a .- b)) for (a, b) in zip(Xdata, d1.X))

    ex = load_exported(model_file)
    E_jl, F_jl, _ = Base.invokelatest(exported_efv, ex, sys, ex.RCUT_MAX)
    dE_lmp = abs(E_jl - E_lmp) / N
    dF_lmp = maximum(norm.(F_jl .- d1.F))

    # ---------------- L. LIVENESS: 100 NVE steps, big enough to reach a GC ---------------
    #
    # Gates A, B and C all evaluate the library ONCE (`run 0`).  So does every accuracy check
    # in this plan.  That is a real hole and Task 6 fell into it: a workspace that the
    # library's own garbage collector reclaimed passed every one of them and then died after
    # roughly one step of a 2048-atom, 100-step run.
    #
    # The arithmetic that sizes this check.  Each site call allocates `Zs` (8n bytes), `Rs`
    # (24n) and the force buffer (24n) = 56n for n neighbours.  The run that exposed the crash
    # was 2048 atoms at n = 201 -- 23 MB per step -- and reached `GC: 1` inside the first step,
    # so the library's first collection lands at roughly 20 MB allocated.  Here the cell is
    # `-var cells 5`, the same one gates A-C use: 500 atoms at n ~ 201 on Cantor is 5.6 MB per
    # step, i.e. **563 MB over 100 steps, about 28x the first-collection threshold**.  Dozens
    # of collections happen with a live workspace in hand.
    #
    # It asserts only that the run COMPLETES and that the energies stay finite: the models here
    # are real fits, but `timestep 0.0` is not used -- the atoms move -- so nothing about the
    # trajectory is a physics gate.  What it catches is the class of fault that has no other
    # detector: a buffer whose lifetime does not survive real use.
    live_input = """
    units metal
    atom_style atomic
    boundary p p p
    read_data $geom
    plugin load $PLUGIN
    pair_style ace
    pair_coeff * * $lib $(sp.species)
    velocity all create 300.0 4928459
    fix nve all nve
    thermo_style custom step pe etotal
    thermo 20
    run 100
    """
    outL = run_lmp(exe, env, live_input, joinpath(work, "liveness.lmp"))
    # Word-bounded, and that matters: a bare `inf` matches "Neighbor list **inf**o ...", which
    # LAMMPS prints on every run -- the first version of this check failed every library for
    # that reason alone.
    live_nonfinite = match(r"(?<![A-Za-z])(nan|inf)(?![A-Za-z])"i, outL)
    live_ok = !occursin("ERROR", outL) && !occursin("LAMMPS_EXIT_NONZERO", outL) &&
              occursin("Loop time of", outL) && live_nonfinite === nothing
    if !live_ok
        error("""[$tag] LIVENESS FAILED -- the library did not survive 100 NVE steps on the
              $(CELLS)^3 cell (~563 MB allocated, ~28x its first-GC threshold).  Gates A/B/C
              evaluate it ONCE and cannot see this; see task-6-report.md section 3.
              LAMMPS output:
              $outL""")
    end
    @printf("[%s] L liveness           : 100 NVE steps completed, energies finite\n", tag)

    # ---------------- B. the library through the Python C API vs Julia, 1e-12 ------------
    penv = copy(env)
    penv["ACE_LIB_PATH"] = lib
    penv["ACE_GEOM"] = geom
    penv["ACE_TYPE_MAP"] = sp.typemap
    penv["PYTHONPATH"] = join(filter(!isempty,
        [joinpath(REPO, "export", "ase-ace", "src"), get(penv, "PYTHONPATH", "")]), ":")
    script = joinpath(REPO, "export", "test", "python", "eval_library.py")
    pyout = try
        read(setenv(`python3 $script`, penv), String)
    catch e
        error("[$tag] python library evaluation failed: $e")
    end
    plines = split(strip(pyout), '\n')
    E_py = parse(Float64, plines[1])
    F_py = [SVector{3,Float64}(parse.(Float64, split(l))...) for l in plines[2:end]]
    dE_py = abs(E_jl - E_py) / N
    dF_py = maximum(norm.(F_jl .- F_py))

    # ---------------- C. two MPI ranks vs one rank, 1e-12 --------------------------------
    out2 = run_lmp(exe, env, lmp_input(dump2), joinpath(work, "gate2.lmp"); ranks = 2)
    mpi_ok = !occursin("ERROR", out2) && !occursin("LAMMPS_EXIT_NONZERO", out2)
    # Loud, never a skip: a gate that passes when the decomposition did not happen is worse
    # than no gate.  `assert_ranks` throws, so this aborts the script rather than recording a
    # green row.  The 1-rank reference run is checked too -- if THAT silently ran on 2 ranks
    # the comparison is equally meaningless.
    mpi_grid = "not-checked"
    if mpi_ok
        assert_ranks(out1, 1)
        mpi_grid = assert_ranks(out2, 2)
        @printf("[%s] 2-rank run confirmed: %s MPI processor grid, 2 procs (from LAMMPS' own output)\n",
                tag, mpi_grid)
    end
    dE_mpi = dF_mpi = NaN            # dE_mpi: per atom, REPORTED only
    rel_mpi = ulp_mpi = abs_mpi = NaN # rel_mpi: relative, GATED; abs_mpi: total, REPORTED
    if mpi_ok
        E2 = parse_ace_energy(out2)
        d2 = read_lammps_dump(dump2)
        abs_mpi = abs(E2 - E_lmp)
        dE_mpi = abs_mpi / N
        dF_mpi = maximum(norm.(d2.F .- d1.F))
        rel_mpi = abs_mpi / abs(E_lmp)          # <- the GATED energy quantity
        ulp_mpi = abs_mpi / eps(abs(E_lmp))
    else
        @warn "[$tag] the 2-rank run did not complete; see $(joinpath(work, "gate2.lmp"))"
    end

    pass = (maxdx == 0.0) && dE_lmp <= TOL_LAMMPS && dF_lmp <= TOL_LAMMPS &&
           dE_py <= TOL_LIB && dF_py <= TOL_LIB &&
           mpi_ok && rel_mpi <= TOL_MPI_E_REL && dF_mpi <= TOL_MPI_F
    # A one-word summary of WHAT failed, carried into every timing row via `gates=`.
    failed_detail = !mpi_ok ? "mpi2_DID_NOT_RUN" :
                    rel_mpi > TOL_MPI_E_REL ? @sprintf("mpi2_energy_FAIL_rel_%.2e_over_%.0e", rel_mpi, TOL_MPI_E_REL) :
                    dF_mpi > TOL_MPI_F ? @sprintf("mpi2_force_FAIL_%.2e_over_%.0e", dF_mpi, TOL_MPI_F) :
                    "mpi2"

    @printf("[%s] N=%d  geometry round-trip max|dx| = %.1e Å\n", tag, N, maxdx)
    @printf("[%s] A LAMMPS vs Julia   : dE/atom = %.3e  max|dF| = %.3e   (tol %.0e) %s\n",
            tag, dE_lmp, dF_lmp, TOL_LAMMPS, (dE_lmp <= TOL_LAMMPS && dF_lmp <= TOL_LAMMPS) ? "PASS" : "FAIL")
    @printf("[%s] B library vs Julia  : dE/atom = %.3e  max|dF| = %.3e   (tol %.0e) %s\n",
            tag, dE_py, dF_py, TOL_LIB, (dE_py <= TOL_LIB && dF_py <= TOL_LIB) ? "PASS" : "FAIL")
    @printf("[%s] C 2 ranks vs 1 rank : |dE|/|E| = %.3e   [GATED, relative, tol %.0e]  %s\n",
            tag, rel_mpi, TOL_MPI_E_REL, (mpi_ok && rel_mpi <= TOL_MPI_E_REL) ? "PASS" : "FAIL")
    @printf("[%s]                        max|dF|  = %.3e eV/Å   [GATED, absolute, tol %.0e]  %s\n",
            tag, dF_mpi, TOL_MPI_F, (mpi_ok && dF_mpi <= TOL_MPI_F) ? "PASS" : "FAIL")
    @printf("[%s]                        |dE|     = %.3e eV total = %.0f ulp of E = %.6g eV   [reported only]\n",
            tag, abs_mpi, ulp_mpi, E_lmp)
    @printf("[%s]                        |dE|/atom= %.3e eV/atom over %d atoms               [reported only]\n",
            tag, dE_mpi, N)
    flush(stdout)

    libsha = bytes2hex(open(sha256, lib))
    open(manifest, "a") do io
        println(io, "# library-level gates -- export/bench/gate_bench_libs.jl, ",
                Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"))
        println(io, "gate_box=$(sp.box) cells=$CELLS natoms=$N")
        println(io, "gate_lammps_exe=$exe")
        println(io, "gate_mpirun=$MPIRUN")
        # Asserted from LAMMPS' own output, not from what mpirun was asked for.  "not-checked"
        # can only appear if the 2-rank run did not complete at all, in which case
        # library_gates is FAIL anyway.
        println(io, "mpi2_decomposition_confirmed=$mpi_grid procs=2   # asserted, not assumed")
        @printf(io, "lammps_vs_julia_tol=%.0e dE_per_atom=%.6e dF=%.6e\n", TOL_LAMMPS, dE_lmp, dF_lmp)
        @printf(io, "library_vs_julia_tol=%.0e dE_per_atom=%.6e dF=%.6e\n", TOL_LIB, dE_py, dF_py)
        @printf(io, "mpi2_vs_mpi1_energy_tol_relative=%.0e dE_relative=%.6e   # GATED\n",
                TOL_MPI_E_REL, rel_mpi)
        @printf(io, "mpi2_vs_mpi1_force_tol_absolute=%.0e dF=%.6e   # GATED\n", TOL_MPI_F, dF_mpi)
        @printf(io, "# reported only, never gated: mpi2 |dE| = %.6e eV total = %.0f ulp of E = %.6g eV; |dE|/atom = %.6e eV/atom\n",
                abs_mpi, ulp_mpi, E_lmp, dE_mpi)
        @printf(io, "geometry_roundtrip_max_dx=%.1e\n", maxdx)
        println(io, "liveness_100_nve_steps=PASS   # ~563 MB allocated, ~28x the library's first-GC threshold")
        println(io, "library_gates=", pass ? "PASS" : "FAIL")
        # bench_parity.sh reads `gates=` and `lib_sha256=` from the manifest, and nothing else.
        # `lib_sha256` is an IDENTITY, not a verdict, so it is written either way -- the
        # verdict travels in `gates=`, which bench_parity.sh prints into every row.  A library
        # whose gates did not all pass is refused unless the caller sets ALLOW_PARTIAL_GATE,
        # and the row then says so.
        println(io, "gates=", pass ? "src,lib,lammps,mpi2,live" : "src,lib,lammps,$(failed_detail)")
        println(io, "lib_sha256=$libsha")
    end
    pass || @error "[$tag] library gates FAILED -- bench_parity.sh will refuse to time it unless ALLOW_PARTIAL_GATE is set"
    results[tag] = (; N, maxdx, dE_lmp, dF_lmp, dE_py, dF_py, rel_mpi, abs_mpi, dE_mpi, dF_mpi, pass, libsha)
end

println("=" ^ 108)
println("LIBRARY GATE SUMMARY")
println("  A: LAMMPS vs Julia            -- dE/atom and max|dF|, both ABSOLUTE, tol 1e-10")
println("  B: library via Python C API vs Julia -- dE/atom and max|dF|, both ABSOLUTE, tol 1e-12")
println("  C: 2 ranks vs 1 rank          -- energy |dE|/|E| RELATIVE tol 1e-13; forces max|dF| ABSOLUTE tol 1e-12")
println("     (C's per-atom and total energy deviations are printed too, but are NOT the gate)")
@printf("%-12s %5s  %11s %11s  %11s %11s  %11s %11s  %11s %11s  %s\n", "tag", "N",
        "A dE/atom", "A dF", "B dE/atom", "B dF", "C dE/|E|*", "C dF*", "C dE/atom", "C |dE| eV", "verdict")
for tag in TAGS
    r = results[tag]
    @printf("%-12s %5d  %11.3e %11.3e  %11.3e %11.3e  %11.3e %11.3e  %11.3e %11.3e  %s\n",
            tag, r.N, r.dE_lmp, r.dF_lmp, r.dE_py, r.dF_py, r.rel_mpi, r.dF_mpi, r.dE_mpi, r.abs_mpi,
            r.pass ? "PASS" : "FAIL")
end
println("  * = the gated quantity")
println("DONE gate_bench_libs.jl")
if !all(results[t].pass for t in TAGS)
    @error "at least one library gate FAILED -- see the summary above; the manifests record it " *
           "and bench_parity.sh will refuse those libraries unless ALLOW_PARTIAL_GATE is set"
    exit(1)
end
