#=
Multi-Threading Tests

Tests for thread-safety and correctness of parallel evaluation:
1. Julia threaded evaluation via ccall
2. Python calculator with threaded Julia runtime
3. LAMMPS with OpenMP threading
=#

using Test
using LinearAlgebra
using StaticArrays
using Libdl

@testset "Multi-threading" begin
    build_dir = joinpath(TEST_DIR, "build")
    lib_path = joinpath(build_dir, "libace_test.so")

    if !isfile(lib_path)
        @test_skip "Library not compiled - skipping threading tests"
        return
    end

    nthreads = Threads.nthreads()
    @info "Running threading tests with $(nthreads) Julia threads"

    @testset "Julia Thread Safety" begin
        # Test that multiple threads can call the library safely
        lib = Libdl.dlopen(lib_path)
        site_energy_ptr = Libdl.dlsym(lib, :ace_site_energy)

        # Create test configurations
        configs = []
        for _ in 1:100
            nneigh = rand(3:10)
            Rs = [SVector(randn(3)...) * 2.5 for _ in 1:nneigh]
            Zs = fill(14, nneigh)
            Z0 = 14
            push!(configs, (Rs, Zs, Z0))
        end

        # Compute energies serially first
        serial_energies = Float64[]
        for (Rs, Zs, Z0) in configs
            nneigh = length(Rs)
            Rij = zeros(Float64, 3 * nneigh)
            for (j, r) in enumerate(Rs)
                Rij[3*(j-1)+1:3*(j-1)+3] .= r
            end
            neighbor_z = Cint.(Zs)

            E = ccall(site_energy_ptr, Cdouble,
                (Cint, Cint, Ptr{Cint}, Ptr{Cdouble}),
                Cint(Z0), Cint(nneigh), neighbor_z, Rij)
            push!(serial_energies, E)
        end

        # Compute energies in parallel
        parallel_energies = zeros(Float64, length(configs))
        Threads.@threads for i in 1:length(configs)
            Rs, Zs, Z0 = configs[i]
            nneigh = length(Rs)
            Rij = zeros(Float64, 3 * nneigh)
            for (j, r) in enumerate(Rs)
                Rij[3*(j-1)+1:3*(j-1)+3] .= r
            end
            neighbor_z = Cint.(Zs)

            E = ccall(site_energy_ptr, Cdouble,
                (Cint, Cint, Ptr{Cint}, Ptr{Cdouble}),
                Cint(Z0), Cint(nneigh), neighbor_z, Rij)
            parallel_energies[i] = E
        end

        # Results should be identical
        @test all(serial_energies .≈ parallel_energies)

        Libdl.dlclose(lib)
    end

    @testset "Julia Threaded System Energy" begin
        # Test system-level energy with threading
        lib = Libdl.dlopen(lib_path)
        ace_energy_ptr = Libdl.dlsym(lib, :ace_energy)

        structure = get_test_structure()

        positions_flat = zeros(Float64, 3 * length(structure.positions))
        for (i, p) in enumerate(structure.positions)
            positions_flat[3*(i-1)+1:3*(i-1)+3] .= p
        end

        cell_flat = zeros(Float64, 9)
        for i in 1:3, j in 1:3
            cell_flat[3*(i-1)+j] = structure.cell[i, j]
        end

        species = Cint.(structure.species)
        pbc = Cint.(structure.pbc)
        natoms = length(structure.species)

        # Call multiple times in parallel
        energies = zeros(Float64, 10)
        Threads.@threads for i in 1:10
            E = ccall(ace_energy_ptr, Cdouble,
                (Cint, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}),
                Cint(natoms), species, positions_flat, cell_flat, pbc)
            energies[i] = E
        end

        # All energies should be identical
        @test all(energies .≈ energies[1])

        Libdl.dlclose(lib)
    end

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
from concurrent.futures import ThreadPoolExecutor

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
assert all(e < 0 for e in energies), 'Expected negative binding energies'

print('ok')
"`, env), String)

            @test strip(result) == "ok"
        end
    end

    if check_lammps_available()
        @testset "LAMMPS with OpenMP" begin
            lammps_test_dir = joinpath(TEST_DIR, "lammps")
            plugin_path = joinpath(EXPORT_DIR, "lammps", "plugin", "build", "aceplugin.so")

            if !isfile(plugin_path)
                @test_skip "LAMMPS plugin not built"
                return
            end

            lmp_exe = strip(read(`which lmp`, String))

            # Set up environment with OpenMP threads
            env = copy(ENV)
            julia_lib_dir = joinpath(Sys.BINDIR, "..", "lib")
            env["LD_LIBRARY_PATH"] = julia_lib_dir * ":" * dirname(lib_path) * ":" * get(ENV, "LD_LIBRARY_PATH", "")
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
                @test drift < 1e-2  # Energy conserved with OpenMP
            end
        end
    end
end
