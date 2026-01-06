#=
Portable Python Calculator Tests

Tests that the relocatable .so files produced by build_deployment work from Python
WITHOUT requiring Julia to be installed on the target system.

The key difference from test_python.jl:
- test_python.jl uses LD_LIBRARY_PATH pointing to Julia's system libraries
- This test uses ONLY the bundled libraries from a deployment package

This simulates what an end-user would experience when using a portable deployment
on a system without Julia installed.
=#

using Test

@testset "Portable Python Calculator" verbose=true begin
    build_dir = joinpath(TEST_DIR, "build")
    lib_path = joinpath(build_dir, "libace_test.so")

    if !isfile(lib_path)
        @test_skip "Library not compiled - skipping portable Python tests"
        return
    end

    # Create a portable deployment directory with bundled Julia runtime
    portable_dir = joinpath(build_dir, "portable_test")
    portable_lib_dir = joinpath(portable_dir, "lib")

    @testset "Create Portable Bundle" begin
        # Clean up any previous test
        isdir(portable_dir) && rm(portable_dir; recursive=true)
        mkpath(portable_lib_dir)

        # Copy the compiled library
        cp(lib_path, joinpath(portable_lib_dir, "libace_test.so"); force=true)
        @test isfile(joinpath(portable_lib_dir, "libace_test.so"))

        # Bundle Julia runtime libraries (same logic as build_deployment.jl)
        julia_lib_dir = joinpath(dirname(Sys.BINDIR), "lib")
        julia_lib_julia_dir = joinpath(julia_lib_dir, "julia")

        # Get list of required libraries from ldd
        ldd_output = try
            read(`ldd $lib_path`, String)
        catch e
            @warn "ldd failed: $e"
            ""
        end

        if !isempty(ldd_output)
            # Libraries to bundle (from Julia installation)
            required_libs = String[]
            for line in split(ldd_output, '\n')
                if contains(line, julia_lib_dir)
                    # Extract library path
                    m = match(r"=> (\S+)", line)
                    if m !== nothing
                        push!(required_libs, m.captures[1])
                    end
                end
            end

            # Also include libjulia.so explicitly
            libjulia = joinpath(julia_lib_dir, "libjulia.so")
            if isfile(libjulia) && !(libjulia in required_libs)
                push!(required_libs, libjulia)
            end

            # Copy libraries
            copied = 0
            for lib in required_libs
                if isfile(lib)
                    dst = joinpath(portable_lib_dir, basename(lib))
                    if !isfile(dst)
                        cp(lib, dst; follow_symlinks=true)
                        copied += 1
                    end
                end
            end

            @info "Bundled $copied Julia runtime libraries into portable deployment"
            @test copied > 0  # Should have bundled at least some libraries
        else
            @warn "Could not run ldd - test may fail on systems without Julia libs"
        end
    end

    # Copy the Python calculator to the portable directory
    python_calc_src = joinpath(EXPORT_DIR, "python", "ace_calculator.py")
    python_calc_dst = joinpath(portable_dir, "ace_calculator.py")
    cp(python_calc_src, python_calc_dst; force=true)

    # Now run Python tests using ONLY the bundled libraries
    # This simulates a system without Julia installed

    @testset "Portable Library Loading" begin
        # Create environment that uses ONLY bundled libs
        env = Dict{String, String}()
        env["LD_LIBRARY_PATH"] = portable_lib_dir  # ONLY bundled libs, not Julia system
        env["PYTHONPATH"] = portable_dir
        env["ACE_LIB_PATH"] = joinpath(portable_lib_dir, "libace_test.so")

        # Preserve minimal required env vars
        for key in ["PATH", "HOME", "USER", "TERM"]
            if haskey(ENV, key)
                env[key] = ENV[key]
            end
        end

        # Test that Python can load the library with only bundled libs
        result = try
            output = read(setenv(`python3 -c "
import os
import ctypes

lib_path = os.environ['ACE_LIB_PATH']
print(f'Loading: {lib_path}', flush=True)

# This should work with only bundled libraries
lib = ctypes.CDLL(lib_path)
print('Library loaded successfully', flush=True)

# Test that we can call a function
lib.ace_get_cutoff.restype = ctypes.c_double
cutoff = lib.ace_get_cutoff()
print(f'Cutoff: {cutoff}', flush=True)

print('ok', flush=True)
"`, env), String)
            # Check if 'ok' appears in the output
            contains(output, "ok") ? "ok" : "failed: $output"
        catch e
            @error "Library loading failed" exception=e
            "error: $e"
        end

        @test result == "ok"
    end

    @testset "Portable Calculator Basic" begin
        env = Dict{String, String}()
        env["LD_LIBRARY_PATH"] = portable_lib_dir
        env["PYTHONPATH"] = portable_dir
        env["ACE_LIB_PATH"] = joinpath(portable_lib_dir, "libace_test.so")
        for key in ["PATH", "HOME", "USER", "TERM"]
            if haskey(ENV, key)
                env[key] = ENV[key]
            end
        end

        result = try
            read(setenv(`python3 -c "
import os
import sys
sys.path.insert(0, os.environ.get('PYTHONPATH', ''))

from ace_calculator import ACECalculator
from ase.build import bulk
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
calc = ACECalculator(lib_path)

# Test with Si diamond
atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = calc

E = atoms.get_potential_energy()
F = atoms.get_forces()
S = atoms.get_stress()

# Sanity checks
assert np.isfinite(E), f'Energy not finite: {E}'
assert np.all(np.isfinite(F)), 'Forces not finite'
assert np.all(np.isfinite(S)), 'Stress not finite'

# Perfect crystal should have near-zero forces
max_force = np.abs(F).max()
assert max_force < 1e-6, f'Max force too large: {max_force}'

print('ok')
"`, env), String)
        catch e
            @error "Calculator test failed" exception=e
            "error: $e"
        end

        @test strip(result) == "ok"
    end

    @testset "Portable MD Energy Conservation" begin
        env = Dict{String, String}()
        env["LD_LIBRARY_PATH"] = portable_lib_dir
        env["PYTHONPATH"] = portable_dir
        env["ACE_LIB_PATH"] = joinpath(portable_lib_dir, "libace_test.so")
        for key in ["PATH", "HOME", "USER", "TERM"]
            if haskey(ENV, key)
                env[key] = ENV[key]
            end
        end

        result = try
            read(setenv(`python3 -c "
import os
import sys
sys.path.insert(0, os.environ.get('PYTHONPATH', ''))

from ace_calculator import ACECalculator
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
calc = ACECalculator(lib_path)

# Create 2x2x2 supercell for MD
atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
atoms.calc = calc

# Initialize velocities at 100 K
MaxwellBoltzmannDistribution(atoms, temperature_K=100)
momenta = atoms.get_momenta()
momenta -= momenta.mean(axis=0)
atoms.set_momenta(momenta)

# Run NVE
dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)

energies = []
for i in range(10):
    dyn.run(5)
    e_kin = atoms.get_kinetic_energy()
    e_pot = atoms.get_potential_energy()
    energies.append(e_kin + e_pot)

energies = np.array(energies)
drift = np.abs(energies[-1] - energies[0])
std = np.std(energies)

# Energy should be conserved
assert drift < 0.1, f'Energy drift too large: {drift}'
assert std < 0.1, f'Energy fluctuation too large: {std}'

print(f'{drift:.6e},{std:.6e}')
"`, env), String)
        catch e
            @error "MD test failed" exception=e
            "error"
        end

        if !startswith(strip(result), "error")
            parts = split(strip(result), ",")
            drift = parse(Float64, parts[1])
            std = parse(Float64, parts[2])
            @test drift < 0.1
            @test std < 0.1
        else
            @test false  # Test failed
        end
    end

    @testset "Portable Finite Difference Forces" begin
        env = Dict{String, String}()
        env["LD_LIBRARY_PATH"] = portable_lib_dir
        env["PYTHONPATH"] = portable_dir
        env["ACE_LIB_PATH"] = joinpath(portable_lib_dir, "libace_test.so")
        for key in ["PATH", "HOME", "USER", "TERM"]
            if haskey(ENV, key)
                env[key] = ENV[key]
            end
        end

        result = try
            read(setenv(`python3 -c "
import os
import sys
sys.path.insert(0, os.environ.get('PYTHONPATH', ''))

from ace_calculator import ACECalculator
from ase.build import bulk
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
calc = ACECalculator(lib_path)

# Create perturbed structure
atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
np.random.seed(42)
atoms.positions += np.random.randn(*atoms.positions.shape) * 0.05
atoms.calc = calc

# Get analytic forces
F_analytic = atoms.get_forces()

# Finite difference for first atom
h = 1e-5
F_fd = np.zeros(3)
for alpha in range(3):
    pos = atoms.positions.copy()

    pos[0, alpha] += h
    atoms.positions = pos
    E_p = atoms.get_potential_energy()

    pos[0, alpha] -= 2*h
    atoms.positions = pos
    E_m = atoms.get_potential_energy()

    F_fd[alpha] = -(E_p - E_m) / (2*h)

    # Restore
    pos[0, alpha] += h
    atoms.positions = pos

max_err = np.max(np.abs(F_analytic[0] - F_fd))
print(f'{max_err:.6e}')
"`, env), String)
        catch e
            @error "FD test failed" exception=e
            "1.0"  # Return a large error on failure
        end

        max_err = parse(Float64, strip(result))
        @test max_err < 1e-2  # FD with h=1e-5 should give ~1e-4 to 1e-3 accuracy
    end

    # Clean up the portable test directory
    @testset "Cleanup" begin
        if isdir(portable_dir)
            rm(portable_dir; recursive=true)
            @test !isdir(portable_dir)
        end
    end
end
