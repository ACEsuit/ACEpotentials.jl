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
#   C. two MPI ranks vs one rank, same library, same geometry                 tol 1e-12
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
const TOL_MPI = 1e-12

const SPEC = Dict(
    "cantor" => (box = "box_cantor.lmp", elements = (:Cr, :Mn, :Fe, :Co, :Ni),
                 species = "Cr Mn Fe Co Ni", typemap = "1:24,2:25,3:26,4:27,5:28"),
    "tial"   => (box = "box_tial.lmp", elements = (:Ti, :Al),
                 species = "Ti Al", typemap = "1:22,2:13"))
model_of(tag) = startswith(tag, "cantor") ? "cantor" : "tial"

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
    dE_mpi = dF_mpi = NaN
    rel_mpi = ulp_mpi = NaN
    if mpi_ok
        E2 = parse_ace_energy(out2)
        d2 = read_lammps_dump(dump2)
        dE_mpi = abs(E2 - E_lmp) / N
        dF_mpi = maximum(norm.(d2.F .- d1.F))
        # REPORTED ONLY, never gated (the same distinction check_export.jl draws for the
        # virial).  The gated quantity is |dE|/natoms in eV/atom -- the convention
        # export/lammps/test/compare_dump.py already uses.  But the TOTAL energy scale differs
        # by 30x between these two models because E0(Ti) = -1586 eV/atom against
        # E0(Cr) ~ -14, so the SAME per-atom gate is ~64x tighter in relative terms on TiAl.
        # Printing the relative deviation and the ulp count is what makes a failure
        # interpretable instead of merely red.
        rel_mpi = abs(E2 - E_lmp) / abs(E_lmp)
        ulp_mpi = abs(E2 - E_lmp) / eps(abs(E_lmp))
    else
        @warn "[$tag] the 2-rank run did not complete; see $(joinpath(work, "gate2.lmp"))"
    end

    pass = (maxdx == 0.0) && dE_lmp <= TOL_LAMMPS && dF_lmp <= TOL_LAMMPS &&
           dE_py <= TOL_LIB && dF_py <= TOL_LIB &&
           mpi_ok && dE_mpi <= TOL_MPI && dF_mpi <= TOL_MPI
    # A one-word summary of WHAT failed, carried into every timing row via `gates=`.
    failed_detail = !mpi_ok ? "mpi2_DID_NOT_RUN" :
                    dE_mpi > TOL_MPI ? @sprintf("mpi2_energy_FAIL_%.2e_over_%.0e", dE_mpi, TOL_MPI) :
                    dF_mpi > TOL_MPI ? @sprintf("mpi2_force_FAIL_%.2e_over_%.0e", dF_mpi, TOL_MPI) :
                    "mpi2"

    @printf("[%s] N=%d  geometry round-trip max|dx| = %.1e Å\n", tag, N, maxdx)
    @printf("[%s] A LAMMPS vs Julia   : dE/atom = %.3e  max|dF| = %.3e   (tol %.0e) %s\n",
            tag, dE_lmp, dF_lmp, TOL_LAMMPS, (dE_lmp <= TOL_LAMMPS && dF_lmp <= TOL_LAMMPS) ? "PASS" : "FAIL")
    @printf("[%s] B library vs Julia  : dE/atom = %.3e  max|dF| = %.3e   (tol %.0e) %s\n",
            tag, dE_py, dF_py, TOL_LIB, (dE_py <= TOL_LIB && dF_py <= TOL_LIB) ? "PASS" : "FAIL")
    @printf("[%s] C 2 ranks vs 1 rank : dE/atom = %.3e  max|dF| = %.3e   (tol %.0e) %s\n",
            tag, dE_mpi, dF_mpi, TOL_MPI, (mpi_ok && dE_mpi <= TOL_MPI && dF_mpi <= TOL_MPI) ? "PASS" : "FAIL")
    @printf("[%s]   (reported only: |dE|/|E| = %.3e = %.0f ulp of E = %.6g eV; forces are intensive and carry no E0)\n",
            tag, rel_mpi, ulp_mpi, E_lmp)
    flush(stdout)

    libsha = bytes2hex(open(sha256, lib))
    open(manifest, "a") do io
        println(io, "# library-level gates -- export/bench/gate_bench_libs.jl, ",
                Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"))
        println(io, "gate_box=$(sp.box) cells=$CELLS natoms=$N")
        println(io, "gate_lammps_exe=$exe")
        @printf(io, "lammps_vs_julia_tol=%.0e dE_per_atom=%.6e dF=%.6e\n", TOL_LAMMPS, dE_lmp, dF_lmp)
        @printf(io, "library_vs_julia_tol=%.0e dE_per_atom=%.6e dF=%.6e\n", TOL_LIB, dE_py, dF_py)
        @printf(io, "mpi2_vs_mpi1_tol=%.0e dE_per_atom=%.6e dF=%.6e\n", TOL_MPI, dE_mpi, dF_mpi)
        @printf(io, "# reported only, never gated: mpi2 |dE|/|E| = %.3e = %.0f ulp of E = %.6g eV\n",
                rel_mpi, ulp_mpi, E_lmp)
        @printf(io, "geometry_roundtrip_max_dx=%.1e\n", maxdx)
        println(io, "library_gates=", pass ? "PASS" : "FAIL")
        # bench_parity.sh reads `gates=` and `lib_sha256=` from the manifest, and nothing else.
        # `lib_sha256` is an IDENTITY, not a verdict, so it is written either way -- the
        # verdict travels in `gates=`, which bench_parity.sh prints into every row.  A library
        # whose gates did not all pass is refused unless the caller sets ALLOW_PARTIAL_GATE,
        # and the row then says so.
        println(io, "gates=", pass ? "src,lib,lammps,mpi2" : "src,lib,lammps,$(failed_detail)")
        println(io, "lib_sha256=$libsha")
    end
    pass || @error "[$tag] library gates FAILED -- bench_parity.sh will refuse to time it unless ALLOW_PARTIAL_GATE is set"
    results[tag] = (; N, maxdx, dE_lmp, dF_lmp, dE_py, dF_py, dE_mpi, dF_mpi, pass, libsha)
end

println("=" ^ 108)
println("LIBRARY GATE SUMMARY  (A: LAMMPS vs Julia 1e-10 | B: library via Python C API vs Julia 1e-12 | C: 2 ranks vs 1 rank 1e-12)")
@printf("%-12s %5s  %11s %11s  %11s %11s  %11s %11s  %s\n", "tag", "N",
        "A dE/atom", "A dF", "B dE/atom", "B dF", "C dE/atom", "C dF", "verdict")
for tag in TAGS
    r = results[tag]
    @printf("%-12s %5d  %11.3e %11.3e  %11.3e %11.3e  %11.3e %11.3e  %s\n",
            tag, r.N, r.dE_lmp, r.dF_lmp, r.dE_py, r.dF_py, r.dE_mpi, r.dF_mpi,
            r.pass ? "PASS" : "FAIL")
end
println("DONE gate_bench_libs.jl")
if !all(results[t].pass for t in TAGS)
    @error "at least one library gate FAILED -- see the summary above; the manifests record it " *
           "and bench_parity.sh will refuse those libraries unless ALLOW_PARTIAL_GATE is set"
    exit(1)
end
