#=
Export Test Suite - Main Test Runner

This file orchestrates all export-related tests for ETACE models:
1. ETACE export functionality (polynomial radial basis)
2. Export accuracy on a small Si model (:polynomial vs the fitted model at 1e-12, plus the
   refusals that replaced the removed spline export)
3. Multi-species model tests
4. Pair-potential export (ETOneBody + ETPairModel + ETACE, Cantor fixture)
4b. AA product DAG (Task 7 / B3), both benchmark models
5. Python calculator integration
6. LAMMPS plugin integration (serial)
7. MPI parallel tests
8. Generator-to-generator parity (OPT-IN: not part of `all`; see below)

Usage:
    julia --project=.. runtests.jl              # Run all available tests
    julia --project=.. runtests.jl etace        # Run ETACE polynomial export tests
    julia --project=.. runtests.jl accuracy     # Run the small-Si accuracy + refusal tests
    julia --project=.. runtests.jl multispecies # Run multi-species tests
    julia --project=.. runtests.jl pair         # Run pair-potential export tests (Cantor fixture)
    julia --project=.. runtests.jl dag          # Run AA-DAG tests (both benchmark models)
    julia --project=.. runtests.jl python       # Run Python tests
    julia --project=.. runtests.jl lammps       # Run LAMMPS tests
    julia --project=.. runtests.jl mpi          # Run MPI tests

    EXPORT_REF_SHA=<sha> julia --project=.. runtests.jl parity
                                                # Generator-to-generator parity at 1e-13
                                                # RELATIVE against the generator at <sha>.
                                                # OPT-IN: never runs under `all`.

GROUPS THAT NO LONGER EXIST, named here rather than left as an absence.  `:hermite_spline`
was removed (`export/bench/FINDINGS_parity.md` §7), and with it three groups:

    hermite           test_hermite_spline_export.jl   -- DELETED with the mode
    hermite_cantor    test_hermite_cantor.jl          -- DELETED with the mode
    hermite_accuracy  test_hermite_accuracy.jl        -- RENAMED to the `accuracy` group
                                                         (test_export_accuracy.jl), which
                                                         keeps its :polynomial gates and adds
                                                         the removal's refusals

A caller who passes `hermite` gets the unknown-selection path, not a silent no-op; and
`ACE_REQUIRE_GROUPS=hermite` FAILS, because no group by that name ever records a status.

THE `parity` GROUP, AND WHY IT IS OPT-IN.  The plan's global constraints bind every
performance step (Tasks 5-7) to the PREVIOUS generator's exported model: "any force differing
by more than 1e-13 relative is a bug, not a speed-up".  test_generator_parity.jl is that gate.
It is not in the default set because it needs (a) both host-local fixtures, (b) a readable git
object for the reference generator, and (c) ~3.5 minutes -- but it IS selectable, it is listed
here, and when it cannot run it reports a visible Broken entry rather than nothing at all.

`ACE_REQUIRE_GROUPS=parity` turns "the parity gate did not run" into a test FAILURE, which is
how a CI job or a task dispatch should ask for it.  There is deliberately no default for
EXPORT_REF_SHA: a parity gate that picks its own reference can compare a change against itself.
=#

using Test
using ACEpotentials
using ACEfit
using ExtXYZ
using StaticArrays
using LinearAlgebra
using JSON

# Test configuration
const TEST_DIR = @__DIR__
const EXPORT_DIR = dirname(TEST_DIR)
const PROJECT_DIR = dirname(EXPORT_DIR)

# Global test artifacts (created once, reused across tests)
const TEST_ARTIFACTS = Dict{String, Any}()

# LD_LIBRARY_PATH construction, LAMMPS executable probing and the LAMMPS data/dump readers,
# shared by the python, lammps and mpi groups.  See its header for K1 and K2.
include(joinpath(TEST_DIR, "lammps_harness.jl"))
# The build-stamp rule, shared verbatim with the generator (export/src/export_ace_model.jl
# includes the same file).  `library_in_step` below needs `export_build_id` whether or not a
# group that loads the generator is selected.
include(joinpath(EXPORT_DIR, "src", "build_stamp.jl"))

"""
    get_test_structure()

Get a simple Si diamond structure for testing.
"""
function get_test_structure()
    # Si diamond conventional cell
    a = 5.43  # Angstrom
    cell = a * [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

    # 8 atoms in conventional cell
    positions = a * [
        SVector(0.00, 0.00, 0.00),
        SVector(0.50, 0.50, 0.00),
        SVector(0.50, 0.00, 0.50),
        SVector(0.00, 0.50, 0.50),
        SVector(0.25, 0.25, 0.25),
        SVector(0.75, 0.75, 0.25),
        SVector(0.75, 0.25, 0.75),
        SVector(0.25, 0.75, 0.75),
    ]

    species = fill(14, 8)  # Si atomic number
    pbc = [true, true, true]

    return (positions=positions, cell=cell, species=species, pbc=pbc)
end

"""
    cantor_fixture_paths()

The two host-local inputs the Cantor fixture needs: the fitted parameters and the first of
the held-out LAMMPS geometries. Neither is part of the repository.
"""
function cantor_fixture_paths()
    return (params = joinpath(PROJECT_DIR, "verify_cantor", "cantor_v010_params.jld2"),
            held = expanduser(joinpath("~", "si-ace", "spike_yace", "cantor", "cantor_1.data")))
end

"""
    check_cantor_fixture_available()

Whether the pair-export test can run here. The test checks the exported model against the
fitted Cantor model, whose saved parameters and held-out geometries are host-local data.

Where they are absent the group is skipped — but NOT silently: it still emits a testset
containing a `@test_skip`, so the summary carries a non-zero broken/skipped count and the
run is visibly distinguishable from one where the pair-parity gate actually executed. A
warning alone would make an absent gate look exactly like a passing gate, which is the CI
blindness this plan exists to remove.
"""
function check_cantor_fixture_available()
    p = cantor_fixture_paths()
    return isfile(p.params) && isfile(p.held)
end

"""
    dag_prereqs() -> (ok::Bool, reason::String)

Whether the `dag` group (test_dag.jl, Task 7 / B3) can run here.  It checks the ported AA-DAG
builder on BOTH benchmark models, so it needs the Cantor fixture and the TiAl order-4
parameters -- the same two host-local data prerequisites as `parity`, but neither
`EXPORT_REF_SHA` nor a resolvable git object, because it compares the generator against the
MODEL rather than against an earlier generator.
"""
function dag_prereqs()
    check_cantor_fixture_available() ||
        return (false, "Cantor fixture data missing")
    tial = joinpath(PROJECT_DIR, "bench_parity", "tial_o4_params.jld2")
    isfile(tial) || return (false, "TiAl parameters missing ($tial)")
    return (true, "")
end

"""
    parity_prereqs() -> (ok::Bool, reason::String)

Whether the `parity` group can run here, and why not if it cannot.  Three separate
prerequisites, each named individually so that a skip line says which one is missing:

  * the Cantor fixture (same host-local data the `pair` group needs);
  * the TiAl order-4 fitted parameters, `bench_parity/tial_o4_params.jld2` (not checked in;
    `export/bench/fit_tial_order4.jl` produces it);
  * `EXPORT_REF_SHA`, which test_generator_parity.jl has no default for, AND a git object
    that actually resolves -- a stale or mistyped SHA must be a named skip here rather than a
    `git show` failure three minutes into the group.

The controller's ruling for Task 5 is that a skip must be VISIBLE: `skip_group` emits a
`@test_skip` (a non-zero Broken count in the summary) and records `:skipped`, which
`report_group_status` turns into a hard failure when `ACE_REQUIRE_GROUPS` names the group.
"""
function parity_prereqs()
    check_cantor_fixture_available() ||
        return (false, "Cantor fixture data missing")
    tial = joinpath(PROJECT_DIR, "bench_parity", "tial_o4_params.jld2")
    isfile(tial) || return (false, "TiAl parameters missing ($tial)")
    sha = strip(get(ENV, "EXPORT_REF_SHA", ""))
    isempty(sha) && return (false, "EXPORT_REF_SHA is not set (this gate has no default)")
    ok = try
        # `sha^{commit}` must be built as a plain String: `^{}` are shell metacharacters
        # that Julia's command literal refuses to interpolate unquoted.
        rev = string(sha, "^{commit}")
        success(pipeline(`git -C $PROJECT_DIR rev-parse --verify --quiet $rev`;
                         stdout = devnull, stderr = devnull))
    catch
        false
    end
    ok || return (false, "EXPORT_REF_SHA=$sha does not resolve to a commit here")
    return (true, "")
end

"""
    check_python_available()

Check if Python with required packages is available.
"""
function check_python_available()
    try
        # Redirect stderr to devnull to suppress import errors
        result = read(pipeline(`python3 -c "import numpy; import ase; print('ok')"`, stderr=devnull), String)
        return strip(result) == "ok"
    catch
        return false
    end
end

"""
    lammps_setup() -> (; exe, env, mpirun, lib_path, plugin_path)

Memoised discovery of everything the LAMMPS groups need. `exe` is the first candidate that
actually **runs** under `env` (see `find_lammps_exe` in `lammps_harness.jl`) and `env` is the
environment it was proven to run under.

K2: this used to be `which lmp`, taken on trust. Where the first `lmp` on PATH is a broken
wrapper (on this host, a pip-installed binary linked against an absent `libmpi.so.12`), an
un-probed path turns the whole group into 1 failure + 4 errors that read like plugin faults,
and no tolerance measured on such a run means anything.
"""
function lammps_setup()
    haskey(TEST_ARTIFACTS, "lammps_setup") && return TEST_ARTIFACTS["lammps_setup"]
    lib_path = joinpath(TEST_DIR, "build", "libace_test.so")
    plugin_path = joinpath(EXPORT_DIR, "lammps", "plugin", "build", "aceplugin.so")
    env0 = ace_runtime_env(dirname(lib_path))
    exe, env = find_lammps_exe(env0)
    # liblammps.so usually sits next to the executable; add it once the exe is known.
    if !isempty(exe)
        env = ace_runtime_env(dirname(lib_path), dirname(exe),
                              split(get(env, "LD_LIBRARY_PATH", ""), ':'; keepempty = false)...)
    end
    mpirun = isempty(exe) ? "" : find_mpirun(exe, env)
    s = (; exe, env, mpirun, lib_path, plugin_path)
    TEST_ARTIFACTS["lammps_setup"] = s
    return s
end

"Whether a LAMMPS executable that actually runs was found."
check_lammps_available() = !isempty(lammps_setup().exe)

"""
    library_build_id(lib_path) -> Union{UInt64, Nothing}

`ace_build_id()` from a compiled library, or `nothing` if it does not export one.

WHY ANY OF THIS EXISTS.  `build/test_etace_model.jl` is REGENERATED by the `etace` group on
every run, while `libace_test.so` is compiled once by CI (or by hand on a dev host).  Nothing
about a `.so` records the source it came from, so the two drift apart silently and every gate
that goes through the compiled library -- the 1e-10 LAMMPS gate, the 1e-12 Python C-API gate,
the two-rank gate, the whole `test_python.jl` group -- then measures a DIFFERENT model from
the one the Julia-side gates measured, and reports healthy-looking numbers for it.  That is
the state this working tree was found in at the start of Task 6: a model file from today
against a library from a week earlier.  `library_in_step` turns that into a hard error at the
top of each library group; skipping instead was considered and rejected, because a skipped
group reads as "no LAMMPS here", which is the one message a stale artefact must not produce.
"""
function library_build_id(lib_path::AbstractString, env = lammps_setup().env)
    isfile(lib_path) || return nothing
    # OUT OF PROCESS, and it has to be.  A juliac `--trim` library embeds its own Julia
    # runtime; `ccall`ing into it from a host Julia process reaches `ijl_adopt_thread` ->
    # `jl_init_threadtls` and ABORTS the whole process with SIGABRT (measured: Julia 1.12.2,
    # signal 6, `jl_init_threadtls at threading.c:324`).  Python's ctypes is already how
    # `test_python.jl` drives these libraries, and a subprocess cannot take the test suite
    # down with it, so the id is read there.
    py = Sys.which("python3")
    py === nothing && return nothing
    script = """
import ctypes, sys
try:
    lib = ctypes.CDLL(sys.argv[1])
    lib.ace_build_id.restype = ctypes.c_ulonglong
    lib.ace_build_id.argtypes = []
    print(lib.ace_build_id())
except AttributeError:
    print("NOSYMBOL")
"""
    out = try
        # `env` carries the LD_LIBRARY_PATH the juliac library needs (libjulia and the
        # libstdc++ Julia bundles in <julia>/lib/julia); without it the dlopen fails.
        strip(read(pipeline(setenv(`$py -c $script $lib_path`, env); stderr = devnull), String))
    catch
        return nothing
    end
    out == "NOSYMBOL" && return nothing
    return tryparse(UInt64, out)
end

"""
    library_in_step() -> (ok::Bool, reason::String)

Whether `build/libace_test.so` was compiled from the `build/test_etace_model.jl` beside it.
See `library_build_id`'s caller notes below for why this matters; `reason` names the fix.
"""
function library_in_step()
    lib = lammps_setup().lib_path
    src = joinpath(TEST_DIR, "build", "test_etace_model.jl")
    isfile(lib) || return (false, "no compiled library at $lib")
    isfile(src) || return (false, "no generated model at $src (run the `etace` group first)")
    want = export_build_id(src)
    want == 0 && return (false, "$(basename(src)) carries no build stamp -- regenerate it")
    Sys.which("python3") === nothing && return (false,
        "cannot read $(basename(lib))'s build id: no python3 on PATH.  (The id must be read " *
        "out of process -- ccalling a juliac library from this Julia process aborts it.)")
    got = library_build_id(lib)
    got === nothing && return (false,
        "$(basename(lib)) exports no ace_build_id(): it predates the build stamp. " *
        "Recompile it from $(basename(src)).")
    got == want && return (true, "build id 0x$(string(want; base = 16)) matches")
    return (false,
        "$(basename(lib)) has build id 0x$(string(got; base = 16)) but $(basename(src)) " *
        "hashes to 0x$(string(want; base = 16)): the library was compiled from a DIFFERENT " *
        "model.  Recompile it (verify_cantor/compile_lib.jl) before gating anything on it.")
end

"Whether an `mpirun` matching that LAMMPS executable's MPI library was found."
check_mpi_available() = !isempty(lammps_setup().mpirun)

# ---------------------------------------------------------------------------------------
# K3 -- a gate that did not EXECUTE must not look like a gate that passed
# ---------------------------------------------------------------------------------------

"""
`GROUP_STATUS[group] = :ran | :skipped`, filled in as `main` walks the groups.

Every guarded group records itself here, and every skip additionally emits a named
`... (SKIPPED: reason)` testset carrying a `@test_skip`, so the summary shows a non-zero
Broken count. Neither of those *fails* a run, which is not enough for CI: on a runner
without the host-local `verify_cantor/` tree the plan's only pair-parity gate would simply
not run, and a green tick would be reported for a check that never happened.

`ACE_REQUIRE_GROUPS` closes that hole. Set it to a comma-separated list of group names (or
to `all`, meaning every group the selection asked for) and the suite ends with a
`required test groups executed` testset that **fails** for each required group whose status
is not `:ran`. The CI workflow sets it on every job; see `.github/workflows/export-ci.yml`.
"""
const GROUP_STATUS = Dict{String, Symbol}()

record_group!(name::AbstractString, status::Symbol) = (GROUP_STATUS[String(name)] = status)

"Record `name` as skipped and emit a visibly-broken testset saying why."
function skip_group(name::AbstractString, reason::AbstractString; details = nothing)
    record_group!(name, :skipped)
    details === nothing ? (@warn "Skipping $name: $reason") : (@warn "Skipping $name: $reason" details)
    @testset "$name (SKIPPED: $reason)" begin
        @test_skip false
    end
    return nothing
end

"Run `f` as group `name`, recording that it executed."
function run_group(f, name::AbstractString)
    record_group!(name, :ran)
    f()
end

"""
    group_required(name) -> Bool

Whether `ACE_REQUIRE_GROUPS` demands that group `name` executes (`all`, or an explicit
mention). A group being *required* is stronger than the group merely running: it also means
that a CHECK INSIDE the group must not quietly `@test_skip` itself away.

`required_check(cond, name, reason)` is the form to use for that. A `lammps` run on a
machine without `ase` used to emit a `@test_skip` for the 1e-12 library-vs-Julia attribution
gate while the group as a whole reported `ran`, so `ACE_REQUIRE_GROUPS=lammps` passed and
nothing said the attribution check had not happened -- the same defect, one level down, as
the dead 1e-6 comparison this task removed.
"""
function group_required(name::AbstractString)
    spec = strip(get(ENV, "ACE_REQUIRE_GROUPS", ""))
    isempty(spec) && return false
    spec == "all" && return true
    return String(name) in strip.(split(spec, ','; keepempty = false))
end

"""
    required_check(cond::Bool, group, reason) -> Bool

`true` if the guarded check can run. Otherwise it registers the absence: a plain
`@test_skip` when the group is not required, and a **failing** `@test` naming the reason when
`ACE_REQUIRE_GROUPS` names the group.
"""
function required_check(cond::Bool, group::AbstractString, reason::AbstractString)
    cond && return true
    if group_required(group)
        @error "required check unavailable in a REQUIRED group" group reason
        @test (reason, :available) == (reason, :unavailable_but_required)
    else
        @test_skip "$group: $reason"
    end
    return false
end

"""
    KNOWN_GROUPS

Every group name `main` can dispatch on.  It exists so that a name that USED to select a
group -- `hermite`, `hermite_accuracy`, `hermite_cantor`, all retired with the
`:hermite_spline` mode -- fails loudly instead of selecting nothing and reporting a clean
run of zero tests.  `ACE_REQUIRE_GROUPS` already catches the CI form of that mistake; this
catches the command-line form.  Add a name here in the same edit that adds its group.
"""
const KNOWN_GROUPS = (:all, :etace, :accuracy, :multispecies, :dag, :pair,
                      :python, :lammps, :mpi, :parity)

# Parse command line args for selective testing
function get_test_selection()
    if length(ARGS) == 0
        return [:all]
    end
    sel = [Symbol(arg) for arg in ARGS]
    bad = [g for g in sel if !(g in KNOWN_GROUPS)]
    isempty(bad) || error("""
        unknown test group(s): $(join(bad, ", ")).
        Known groups: $(join(KNOWN_GROUPS, ", ")).
        `hermite`, `hermite_accuracy` and `hermite_cantor` were retired with the
        :hermite_spline radial mode (export/bench/FINDINGS_parity.md §7); the surviving
        :polynomial half of `hermite_accuracy` is now the `accuracy` group.""")
    return sel
end

function should_run_test(selection, test_name)
    return :all in selection || test_name in selection
end

# Main test execution
function main()
    selection = get_test_selection()

    @info "ACE Export Test Suite (ETACE)"
    @info "=============================="
    @info "Test selection: $selection"
    @info "Cantor fixture available: $(check_cantor_fixture_available())"
    @info "Parity gate runnable: $(parity_prereqs()[1])  $(parity_prereqs()[2])"
    @info "Python available: $(check_python_available())"
    @info "LAMMPS available: $(check_lammps_available())"
    @info "MPI available: $(check_mpi_available())"
    @info ""

    cantor_missing() = [f for f in values(cantor_fixture_paths()) if !isfile(f)]

    @testset "ACE Export Tests" verbose=true begin
        # ETACE export tests (polynomial radial basis)
        if should_run_test(selection, :etace) || should_run_test(selection, :all)
            run_group("etace") do
                @info "Running ETACE export tests (polynomial)..."
                include(joinpath(TEST_DIR, "test_etace_export.jl"))
            end
        end

        # Export accuracy on a small randomly-parameterised Si model.
        #
        # test_export_accuracy.jl states which reference each of its numbers uses
        # (:polynomial gated against the FITTED model at 1e-12).  It was a standalone script
        # that nothing ran -- so its tolerances were never enforced by anything -- and it
        # was the `hermite_accuracy` group until the spline mode was removed.  It also
        # carries the refusals that removal introduced: a splinified model, and the
        # :hermite_spline keyword.
        if should_run_test(selection, :accuracy) || should_run_test(selection, :all)
            run_group("accuracy") do
                @info "Running export accuracy tests (:polynomial + refusals)..."
                include(joinpath(TEST_DIR, "test_export_accuracy.jl"))
            end
        end

        # Multi-species tests
        if should_run_test(selection, :multispecies) || should_run_test(selection, :all)
            run_group("multispecies") do
                @info "Running multi-species export tests..."
                include(joinpath(TEST_DIR, "test_multispecies.jl"))
            end
        end

        # AA product DAG (Task 7 / B3).  Structural + numeric checks of the ported DAG
        # builder on BOTH benchmark models, so it needs the same host-local fixtures the
        # `pair` and `parity` groups do (and, unlike `parity`, no git object and no
        # EXPORT_REF_SHA -- it checks the generator against itself and against the model).
        if should_run_test(selection, :dag) || should_run_test(selection, :all)
            ok, why = dag_prereqs()
            if ok
                run_group("dag") do
                    @info "Running AA-DAG tests..."
                    include(joinpath(TEST_DIR, "test_dag.jl"))
                end
            else
                skip_group("dag", why)
            end
        end

        # Pair-potential export tests (ETOneBody + ETPairModel + ETACE stack)
        if should_run_test(selection, :pair) || should_run_test(selection, :all)
            if check_cantor_fixture_available()
                run_group("pair") do
                    @info "Running pair-potential export tests..."
                    include(joinpath(TEST_DIR, "test_pair_export.jl"))
                end
            else
                skip_group("pair", "Cantor fixture data missing"; details = cantor_missing())
            end
        end

        # Python tests
        if should_run_test(selection, :python) || should_run_test(selection, :all)
            if check_python_available()
                _lib_ok, _lib_why = library_in_step()
                _lib_ok || error("python group: $_lib_why")
                @info "Compiled library in step with its source: $_lib_why"
                run_group("python") do
                    @info "Running Python tests..."
                    include(joinpath(TEST_DIR, "test_python.jl"))
                end
            else
                skip_group("python", "python3 with numpy and ase not available")
            end
        end

        # LAMMPS tests (serial)
        if should_run_test(selection, :lammps) || should_run_test(selection, :all)
            if check_lammps_available()
                _lib_ok, _lib_why = library_in_step()
                _lib_ok || error("lammps group: $_lib_why")
                @info "Compiled library in step with its source: $_lib_why"
                run_group("lammps") do
                    @info "Running LAMMPS tests (serial)..."
                    include(joinpath(TEST_DIR, "test_lammps.jl"))
                end
            else
                skip_group("lammps", "no LAMMPS executable that runs was found")
            end
        end

        # MPI tests
        if should_run_test(selection, :mpi) || should_run_test(selection, :all)
            if check_mpi_available() && check_lammps_available()
                _lib_ok, _lib_why = library_in_step()
                _lib_ok || error("mpi group: $_lib_why")
                run_group("mpi") do
                    @info "Running MPI tests..."
                    include(joinpath(TEST_DIR, "test_mpi.jl"))
                end
            else
                skip_group("mpi", "no matching mpirun / LAMMPS executable was found")
            end
        end

        # Generator-to-generator parity (Tasks 5-7).  OPT-IN: deliberately NOT part of
        # `:all` -- it needs a git object and ~3.5 minutes -- but it is selectable, and when
        # it is selected and cannot run, the skip is visible in the summary.
        # NOTE: `:parity in selection`, deliberately NOT `should_run_test(...)` -- that helper
        # returns true for `:all`, which would pull this group into the default run (adding
        # ~3.5 minutes, and a Broken entry in every default summary on a host without
        # EXPORT_REF_SHA set).  The group is opt-in: it runs only when named.
        if :parity in selection
            ok, why = parity_prereqs()
            if ok
                run_group("parity") do
                    @info "Running generator-to-generator parity tests (EXPORT_REF_SHA=$(ENV["EXPORT_REF_SHA"]))..."
                    include(joinpath(TEST_DIR, "test_generator_parity.jl"))
                end
            else
                skip_group("parity", why)
            end
        end

        # K3: turn "the gate never ran" into a test failure when CI asks for it.
        report_group_status(selection)
    end

    @info "Test suite completed!"
end

"""
    report_group_status(selection)

Print the status of every group this run touched, then -- if `ACE_REQUIRE_GROUPS` is set --
assert that each required group actually executed.

`ACE_REQUIRE_GROUPS=all` requires every group that was selected and reached a decision;
otherwise it is a comma-separated list of group names (`etace`, `accuracy`,
`multispecies`, `dag`, `pair`, `parity`, `python`, `lammps`, `mpi`). A required group that
was skipped, or that was never reached because the selection excluded it, fails.
"""
function report_group_status(selection)
    println("\nTest group execution status:")
    for g in sort(collect(keys(GROUP_STATUS)))
        println("    $(rpad(g, 16)) $(GROUP_STATUS[g] == :ran ? "ran" : "SKIPPED")")
    end
    # Machine-readable form, one line per group, for a CI step to grep.  Kept deliberately
    # simple and stable: `.github/workflows/export-ci.yml` parses exactly this.
    for g in sort(collect(keys(GROUP_STATUS)))
        println("ACE_GROUP_STATUS $g=$(GROUP_STATUS[g] == :ran ? "ran" : "skipped")")
    end
    spec = strip(get(ENV, "ACE_REQUIRE_GROUPS", ""))
    if isempty(spec)
        println("    (ACE_REQUIRE_GROUPS is unset: a SKIPPED group above is NOT a failure)")
        flush(stdout)
        return nothing
    end
    required = spec == "all" ? sort(collect(keys(GROUP_STATUS))) :
               String.(split(spec, ','; keepempty = false))
    println("    ACE_REQUIRE_GROUPS = $spec -> requiring: $(join(required, ", "))")
    flush(stdout)
    @testset "required test groups executed" begin
        for g in required
            st = get(GROUP_STATUS, strip(g), :not_selected)
            @test (g, st) == (g, :ran)
        end
    end
    return nothing
end

# Run tests
main()
