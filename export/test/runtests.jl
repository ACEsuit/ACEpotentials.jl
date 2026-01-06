#=
Export Test Suite - Main Test Runner

This file orchestrates all export-related tests:
1. Julia export functionality (model export, compilation, evaluation)
2. ETACE export functionality (ETACE model export with solid harmonics)
3. Python calculator integration
4. LAMMPS plugin integration (serial)
5. Multi-species model tests
6. MPI parallel tests

Usage:
    julia --project=.. runtests.jl              # Run all available tests
    julia --project=.. runtests.jl export       # Run only ACEModel export tests
    julia --project=.. runtests.jl etace        # Run only ETACE export tests
    julia --project=.. runtests.jl python       # Run only Python tests
    julia --project=.. runtests.jl portable     # Run only portable Python tests
    julia --project=.. runtests.jl lammps       # Run only LAMMPS tests
    julia --project=.. runtests.jl multispecies # Run only multi-species tests
    julia --project=.. runtests.jl mpi          # Run only MPI tests
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

"""
    setup_test_model()

Fit a small Si model for testing. Returns (potential, test_data).
Caches the result for reuse across test files.
"""
function setup_test_model()
    if haskey(TEST_ARTIFACTS, "potential")
        return TEST_ARTIFACTS["potential"], TEST_ARTIFACTS["test_data"]
    end

    @info "Setting up test model (fitting Si potential)..."

    # Load training data using example_dataset
    dataset = ACEpotentials.example_dataset("Si_tiny")
    data = dataset.train

    # Create small model for fast testing
    model = ACEpotentials.ace1_model(
        elements = [:Si],
        order = 2,
        totaldegree = 6,
        rcut = 5.5,
    )

    # Data keys matching Si_tiny dataset
    data_keys = (
        energy_key = "dft_energy",
        force_key = "dft_force",
        virial_key = "dft_virial",
    )

    weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0))

    # Fit model
    ACEpotentials.acefit!(data, model;
        data_keys...,
        weights = weights,
        solver = ACEfit.BLR(),
    )

    # Get the fitted potential
    potential = model

    # Store for reuse
    TEST_ARTIFACTS["potential"] = potential
    TEST_ARTIFACTS["test_data"] = data

    @info "Test model ready"

    return potential, data
end

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
    check_lammps_available()

Check if LAMMPS is available.
"""
function check_lammps_available()
    try
        result = read(`which lmp`, String)
        return !isempty(strip(result))
    catch
        return false
    end
end

"""
    check_mpi_available()

Check if MPI is available.
"""
function check_mpi_available()
    try
        result = read(`which mpirun`, String)
        return !isempty(strip(result))
    catch
        return false
    end
end

"""
    get_compiled_library()

Get path to compiled test library. Compiles if not exists.
"""
function get_compiled_library()
    if haskey(TEST_ARTIFACTS, "lib_path")
        return TEST_ARTIFACTS["lib_path"]
    end

    # Set up model first
    potential, _ = setup_test_model()

    # Export and compile
    build_dir = joinpath(TEST_DIR, "build")
    mkpath(build_dir)

    @info "Exporting model to Julia code..."
    include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))
    model_file = joinpath(build_dir, "test_ace_model.jl")
    export_ace_model(potential, model_file; for_library=true)

    @info "Compiling to shared library..."
    lib_path = joinpath(build_dir, "libace_test.so")

    # Run juliac
    juliac_cmd = `julia --project=$(EXPORT_DIR) -e "
        using PackageCompiler
        PackageCompiler.juliac(
            \"$(model_file)\",
            \"$(lib_path)\";
            compile_library=true,
            trim=true,
            verbose=true
        )
    "`

    # Alternative: use juliac directly if available
    # juliac_cmd = `juliac --output-lib $(lib_path) --trim=safe --project=$(EXPORT_DIR) $(model_file)`

    run(juliac_cmd)

    if !isfile(lib_path)
        error("Compilation failed: $lib_path not created")
    end

    TEST_ARTIFACTS["lib_path"] = lib_path
    TEST_ARTIFACTS["model_file"] = model_file

    return lib_path
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

    @info "ACE Export Test Suite"
    @info "====================="
    @info "Test selection: $selection"
    @info "Python available: $(check_python_available())"
    @info "LAMMPS available: $(check_lammps_available())"
    @info "MPI available: $(check_mpi_available())"
    @info ""

    @testset "ACE Export Tests" verbose=true begin
        # Julia export tests (ACEModel)
        if should_run_test(selection, :export) || should_run_test(selection, :all)
            @info "Running Julia export tests (ACEModel)..."
            include(joinpath(TEST_DIR, "test_export.jl"))
        end

        # ETACE export tests
        if should_run_test(selection, :etace) || should_run_test(selection, :all)
            @info "Running ETACE export tests..."
            include(joinpath(TEST_DIR, "test_etace_export.jl"))
        end

        # Python tests
        if should_run_test(selection, :python) || should_run_test(selection, :all)
            if check_python_available()
                @info "Running Python tests..."
                include(joinpath(TEST_DIR, "test_python.jl"))
            else
                @warn "Skipping Python tests (Python/numpy/ase not available)"
            end
        end

        # Portable Python tests (validates relocatable deployment)
        if should_run_test(selection, :portable) || should_run_test(selection, :all)
            if check_python_available()
                @info "Running portable Python tests..."
                include(joinpath(TEST_DIR, "test_portable_python.jl"))
            else
                @warn "Skipping portable Python tests (Python/numpy/ase not available)"
            end
        end

        # LAMMPS tests (serial)
        if should_run_test(selection, :lammps) || should_run_test(selection, :all)
            if check_lammps_available()
                @info "Running LAMMPS tests (serial)..."
                include(joinpath(TEST_DIR, "test_lammps.jl"))
            else
                @warn "Skipping LAMMPS tests (lmp not found)"
            end
        end

        # Multi-species tests
        if should_run_test(selection, :multispecies) || should_run_test(selection, :all)
            @info "Running multi-species export tests..."
            include(joinpath(TEST_DIR, "test_multispecies.jl"))
        end

        # MPI tests
        if should_run_test(selection, :mpi) || should_run_test(selection, :all)
            if check_mpi_available() && check_lammps_available()
                @info "Running MPI tests..."
                include(joinpath(TEST_DIR, "test_mpi.jl"))
            else
                @warn "Skipping MPI tests (mpirun or lmp not found)"
            end
        end
    end

    @info "Test suite completed!"
end

# Run tests
main()
