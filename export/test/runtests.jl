#=
Export Test Suite - Main Test Runner

This file orchestrates all export-related tests for ETACE models:
1. ETACE export functionality (polynomial radial basis)
2. Hermite spline export (machine-precision splined radial basis)
3. Multi-species model tests
4. Python calculator integration
5. LAMMPS plugin integration (serial)
6. MPI parallel tests

Usage:
    julia --project=.. runtests.jl              # Run all available tests
    julia --project=.. runtests.jl etace        # Run ETACE polynomial export tests
    julia --project=.. runtests.jl hermite      # Run Hermite spline export tests
    julia --project=.. runtests.jl multispecies # Run multi-species tests
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
    @info "Python available: $(check_python_available())"
    @info "LAMMPS available: $(check_lammps_available())"
    @info "MPI available: $(check_mpi_available())"
    @info ""

    @testset "ACE Export Tests" verbose=true begin
        # ETACE export tests (polynomial radial basis)
        if should_run_test(selection, :etace) || should_run_test(selection, :all)
            @info "Running ETACE export tests (polynomial)..."
            include(joinpath(TEST_DIR, "test_etace_export.jl"))
        end

        # Hermite spline export tests
        if should_run_test(selection, :hermite) || should_run_test(selection, :all)
            @info "Running Hermite spline export tests..."
            include(joinpath(TEST_DIR, "test_hermite_spline_export.jl"))
        end

        # Multi-species tests
        if should_run_test(selection, :multispecies) || should_run_test(selection, :all)
            @info "Running multi-species export tests..."
            include(joinpath(TEST_DIR, "test_multispecies.jl"))
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

        # LAMMPS tests (serial)
        if should_run_test(selection, :lammps) || should_run_test(selection, :all)
            if check_lammps_available()
                @info "Running LAMMPS tests (serial)..."
                include(joinpath(TEST_DIR, "test_lammps.jl"))
            else
                @warn "Skipping LAMMPS tests (lmp not found)"
            end
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
