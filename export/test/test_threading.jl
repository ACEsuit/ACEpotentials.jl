#=
Multi-Threading Tests

Tests for thread-safety and correctness of parallel evaluation:
1. LAMMPS with OpenMP threading

NOTE: Julia ccall tests are disabled because loading a juliac-compiled
library into a running Julia process causes threading conflicts
(jl_init_threadtls / ijl_adopt_thread).
=#

using Test
using LinearAlgebra
using StaticArrays
using Libdl

@testset "Multi-threading" verbose=true begin
    build_dir = joinpath(TEST_DIR, "build")
    lib_path = joinpath(build_dir, "libace_test.so")

    if !isfile(lib_path)
        @test_skip "Library not compiled - skipping threading tests"
    else
        nthreads = Threads.nthreads()
        @info "Running threading tests with $(nthreads) Julia threads"

        # Python threading test (only if Python available)
        if check_python_available()
            @testset "Python with Threading" begin
                python_calc_dir = joinpath(EXPORT_DIR, "python")

                env = copy(ENV)
                env["ACE_LIB_PATH"] = lib_path
                env["PYTHONPATH"] = python_calc_dir * ":" * get(ENV, "PYTHONPATH", "")

                julia_lib_dir = joinpath(Sys.BINDIR, "..", "lib")
                env["LD_LIBRARY_PATH"] = julia_lib_dir * ":" * get(ENV, "LD_LIBRARY_PATH", "")

                # Test Python multiprocessing with Julia library
                result = read(setenv(`python3 -c "
import os
import sys
sys.path.insert(0, os.environ.get('PYTHONPATH', '').split(':')[0])

from ace_calculator import ACECalculator
from ase.build import bulk
import numpy as np

lib_path = os.environ['ACE_LIB_PATH']
calc = ACECalculator(lib_path)

# Create multiple structures
structures = []
for i in range(10):
    atoms = bulk('Si', 'diamond', a=5.43 + i*0.01)
    atoms.calc = calc
    structures.append(atoms)

# Compute energies (single-threaded in Python, but tests Julia runtime)
energies = [atoms.get_potential_energy() for atoms in structures]

# Verify results are reasonable
assert all(np.isfinite(energies)), 'Non-finite energies'

print('ok')
"`, env), String)

                @test strip(result) == "ok"
            end
        end

        # LAMMPS OpenMP test
        if check_lammps_available()
            @testset "LAMMPS with OpenMP" begin
                lammps_test_dir = joinpath(TEST_DIR, "lammps")
                plugin_path = joinpath(EXPORT_DIR, "lammps", "plugin", "build", "aceplugin.so")

                if !isfile(plugin_path)
                    @test_skip "LAMMPS plugin not built"
                else
                    # Find LAMMPS source directory (needed for executable and library path)
                    lammps_src = get(ENV, "LAMMPS_SRC", "")

                    # Find LAMMPS executable (prefer build directory if LAMMPS_SRC is set)
                    lmp_exe = ""
                    if !isempty(lammps_src) && isdir(lammps_src)
                        build_lmp = joinpath(dirname(lammps_src), "build", "lmp")
                        if isfile(build_lmp)
                            lmp_exe = build_lmp
                        end
                    end
                    # Fall back to system lmp
                    if isempty(lmp_exe)
                        lmp_exe = try
                            strip(read(`which lmp`, String))
                        catch
                            ""
                        end
                    end

                    if isempty(lmp_exe) || !isfile(lmp_exe)
                        @test_skip "LAMMPS not found"
                    else
                        # Set up environment with OpenMP threads
                        env = copy(ENV)
                        julia_lib_dir = joinpath(Sys.BINDIR, "..", "lib")

                        # Find LAMMPS library directory
                        lammps_lib_dir = ""
                        if !isempty(lammps_src) && isdir(lammps_src)
                            lammps_build = joinpath(dirname(lammps_src), "build")
                            if isdir(lammps_build) && isfile(joinpath(lammps_build, "liblammps.so"))
                                lammps_lib_dir = lammps_build
                            end
                        end

                        # Prepend required paths to existing LD_LIBRARY_PATH
                        extra_paths = join(filter(!isempty, [
                            julia_lib_dir,
                            dirname(lib_path),
                            lammps_lib_dir,
                        ]), ":")
                        existing_ld_path = get(ENV, "LD_LIBRARY_PATH", "")
                        env["LD_LIBRARY_PATH"] = isempty(existing_ld_path) ? extra_paths : "$extra_paths:$existing_ld_path"
                        env["OMP_NUM_THREADS"] = "4"

                        # Run LAMMPS with OpenMP
                        test_input = """
                        units metal
                        atom_style atomic
                        boundary p p p

                        lattice diamond 5.43
                        region box block 0 2 0 2 0 2
                        create_box 1 box
                        create_atoms 1 box
                        mass 1 28.0855

                        plugin load $(plugin_path)
                        pair_style ace
                        pair_coeff * * $(lib_path) Si

                        velocity all create 100.0 42
                        fix nve all nve

                        thermo_style custom step pe ke etotal
                        thermo 10

                        run 50
                        """

                        input_file = joinpath(lammps_test_dir, "test_omp.lmp")
                        mkpath(lammps_test_dir)
                        write(input_file, test_input)

                        output = try
                            read(setenv(`$(lmp_exe) -in $(input_file)`, env), String)
                        catch e
                            "error: $e"
                        end

                        @test !occursin("ERROR", output)
                        @test occursin("Loop time", output) || occursin("Total wall time", output)

                        # Extract and verify energies
                        lines = split(output, "\n")
                        energies = Float64[]

                        in_thermo = false
                        for line in lines
                            if occursin("Step", line) && occursin("TotEng", line)
                                in_thermo = true
                                continue
                            end
                            if in_thermo
                                parts = split(strip(line))
                                if length(parts) >= 4 && tryparse(Int, parts[1]) !== nothing
                                    etotal = parse(Float64, parts[4])
                                    push!(energies, etotal)
                                elseif occursin("Loop", line)
                                    break
                                end
                            end
                        end

                        if length(energies) >= 5
                            drift = abs(energies[end] - energies[1])
                            # Note: Test model is a small, quickly-fitted potential for testing
                            # infrastructure. Real production models should have better conservation.
                            @test drift < 0.1  # Energy conserved with OpenMP (relaxed for test model)
                        end
                    end
                end
            end
        end
    end
end
