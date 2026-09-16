#=
Export Test Suite - Main Test Runner

This file orchestrates all export-related tests for ETACE models:
1. ETACE export functionality (polynomial radial basis)
2. Hermite spline export (approximate splined radial basis; incl. the fitted Cantor model)
3. Multi-species model tests
4. Pair-potential export (ETOneBody + ETPairModel + ETACE, Cantor fixture)
5. Python calculator integration
6. LAMMPS plugin integration (serial)
7. MPI parallel tests

Usage:
    julia --project=.. runtests.jl              # Run all available tests
    julia --project=.. runtests.jl etace        # Run ETACE polynomial export tests
    julia --project=.. runtests.jl hermite      # Run Hermite spline export tests
    julia --project=.. runtests.jl multispecies # Run multi-species tests
    julia --project=.. runtests.jl pair         # Run pair-potential export tests (Cantor fixture)
    julia --project=.. runtests.jl python       # Run Python tests
    julia --project=.. runtests.jl lammps       # Run LAMMPS tests
    julia --project=.. runtests.jl mpi          # Run MPI tests
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

# Parse command line args for selective testing
function get_test_selection()
    if length(ARGS) == 0
        return [:all]
    else
        return [Symbol(arg) for arg in ARGS]
    end
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

        # Hermite spline export tests
        if should_run_test(selection, :hermite) || should_run_test(selection, :all)
            run_group("hermite") do
                @info "Running Hermite spline export tests..."
                include(joinpath(TEST_DIR, "test_hermite_spline_export.jl"))
            end

            # test_hermite_accuracy.jl states which reference each of its numbers uses
            # (:hermite_spline gated against the SPLINIFIED model at 1e-8, its error against
            # the FITTED model reported only; :polynomial gated against the FITTED model at
            # 1e-12).  It was a standalone script that nothing ran -- so its tolerances were
            # never enforced by anything.  It is part of the `hermite` group now.
            run_group("hermite_accuracy") do
                @info "Running Hermite/polynomial export accuracy tests..."
                include(joinpath(TEST_DIR, "test_hermite_accuracy.jl"))
            end

            # Hermite export of the fitted multi-species Cantor model, gated against the
            # SPLINIFIED stack.  Guarded exactly like the pair group: the fitted parameters
            # and the held-out geometries are host-local, untracked data, and an absent gate
            # must be visible in the summary rather than look like a pass.
            if check_cantor_fixture_available()
                run_group("hermite_cantor") do
                    @info "Running Cantor Hermite export tests..."
                    include(joinpath(TEST_DIR, "test_hermite_cantor.jl"))
                end
            else
                skip_group("hermite_cantor", "Cantor fixture data missing";
                           details = cantor_missing())
            end
        end

        # Multi-species tests
        if should_run_test(selection, :multispecies) || should_run_test(selection, :all)
            run_group("multispecies") do
                @info "Running multi-species export tests..."
                include(joinpath(TEST_DIR, "test_multispecies.jl"))
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
                run_group("mpi") do
                    @info "Running MPI tests..."
                    include(joinpath(TEST_DIR, "test_mpi.jl"))
                end
            else
                skip_group("mpi", "no matching mpirun / LAMMPS executable was found")
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
otherwise it is a comma-separated list of group names (`etace`, `hermite`,
`hermite_cantor`, `multispecies`, `pair`, `python`, `lammps`, `mpi`). A required group that
was skipped, or that was never reached because the selection excluded it, fails.
"""
function report_group_status(selection)
    println("\nTest group execution status:")
    for g in sort(collect(keys(GROUP_STATUS)))
        println("    $(rpad(g, 16)) $(GROUP_STATUS[g] == :ran ? "ran" : "SKIPPED")")
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
