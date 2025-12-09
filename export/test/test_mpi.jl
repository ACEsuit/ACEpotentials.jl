#=
MPI Parallelization Tests

Tests for LAMMPS MPI parallel execution:
1. Domain decomposition correctness
2. Energy/force consistency between serial and parallel
3. Ghost atom handling
=#

using Test

@testset "MPI Parallelization" begin
    build_dir = joinpath(TEST_DIR, "build")
    lib_path = joinpath(build_dir, "libace_test.so")
    lammps_test_dir = joinpath(TEST_DIR, "lammps")
    plugin_path = joinpath(EXPORT_DIR, "lammps", "plugin", "build", "aceplugin.so")

    if !isfile(lib_path)
        @test_skip "ACE library not compiled"
        return
    end

    if !isfile(plugin_path)
        @test_skip "LAMMPS plugin not built"
        return
    end

    # Check for MPI and LAMMPS
    mpirun_exe = try
        strip(read(`which mpirun`, String))
    catch
        ""
    end

    lmp_exe = try
        strip(read(`which lmp`, String))
    catch
        ""
    end

    if isempty(mpirun_exe) || isempty(lmp_exe)
        @test_skip "MPI or LAMMPS not available"
        return
    end

    # Set up environment
    env = copy(ENV)
    julia_lib_dir = joinpath(Sys.BINDIR, "..", "lib")
    env["LD_LIBRARY_PATH"] = join([
        julia_lib_dir,
        dirname(lib_path),
        get(ENV, "LD_LIBRARY_PATH", "")
    ], ":")

    mkpath(lammps_test_dir)

    @testset "MPI Energy Consistency" begin
        # Compare serial vs MPI parallel energy
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

        thermo_style custom step pe
        run 0
        """

        input_file = joinpath(lammps_test_dir, "test_mpi_energy.lmp")
        write(input_file, test_input)

        # Run serial
        output_serial = read(setenv(`$(lmp_exe) -in $(input_file)`, env), String)

        # Run with 4 MPI ranks
        output_mpi = try
            read(setenv(`mpirun -np 4 --oversubscribe $(lmp_exe) -in $(input_file)`, env), String)
        catch
            # Try without --oversubscribe
            read(setenv(`mpirun -np 4 $(lmp_exe) -in $(input_file)`, env), String)
        end

        # Extract energies
        function extract_energy(output)
            lines = split(output, "\n")
            for (i, line) in enumerate(lines)
                if occursin("Step", line) && occursin("PotEng", line)
                    if i < length(lines)
                        parts = split(strip(lines[i+1]))
                        if length(parts) >= 2
                            return parse(Float64, parts[2])
                        end
                    end
                end
            end
            return nothing
        end

        E_serial = extract_energy(output_serial)
        E_mpi = extract_energy(output_mpi)

        @test E_serial !== nothing
        @test E_mpi !== nothing
        @test E_serial ≈ E_mpi rtol=1e-10
    end

    @testset "MPI Force Consistency" begin
        # Dump forces from serial and MPI runs, compare
        test_input = """
        units metal
        atom_style atomic
        boundary p p p

        lattice diamond 5.43
        region box block 0 2 0 2 0 2
        create_box 1 box
        create_atoms 1 box
        mass 1 28.0855

        # Small perturbation for non-zero forces
        displace_atoms all random 0.01 0.01 0.01 42

        plugin load $(plugin_path)
        pair_style ace
        pair_coeff * * $(lib_path) Si

        dump forces all custom 1 DUMPFILE id type fx fy fz
        dump_modify forces sort id format float %20.12e

        run 0
        """

        # Serial
        input_serial = replace(test_input, "DUMPFILE" => joinpath(lammps_test_dir, "forces_serial.dump"))
        write(joinpath(lammps_test_dir, "test_mpi_forces_serial.lmp"), input_serial)
        run(setenv(`$(lmp_exe) -in $(joinpath(lammps_test_dir, "test_mpi_forces_serial.lmp"))`, env))

        # MPI
        input_mpi = replace(test_input, "DUMPFILE" => joinpath(lammps_test_dir, "forces_mpi.dump"))
        write(joinpath(lammps_test_dir, "test_mpi_forces_mpi.lmp"), input_mpi)
        try
            run(setenv(`mpirun -np 4 --oversubscribe $(lmp_exe) -in $(joinpath(lammps_test_dir, "test_mpi_forces_mpi.lmp"))`, env))
        catch
            run(setenv(`mpirun -np 4 $(lmp_exe) -in $(joinpath(lammps_test_dir, "test_mpi_forces_mpi.lmp"))`, env))
        end

        # Read and compare forces
        function read_forces(filename)
            lines = readlines(filename)
            natoms = parse(Int, lines[4])
            forces = zeros(natoms, 3)
            for i in 1:natoms
                parts = split(strip(lines[9 + i - 1]))
                id = parse(Int, parts[1])
                forces[id, 1] = parse(Float64, parts[3])
                forces[id, 2] = parse(Float64, parts[4])
                forces[id, 3] = parse(Float64, parts[5])
            end
            return forces
        end

        F_serial = read_forces(joinpath(lammps_test_dir, "forces_serial.dump"))
        F_mpi = read_forces(joinpath(lammps_test_dir, "forces_mpi.dump"))

        max_diff = maximum(abs.(F_serial - F_mpi))
        @test max_diff < 1e-10
    end

    @testset "Domain Decomposition" begin
        # Test larger system with domain decomposition
        test_input = """
        units metal
        atom_style atomic
        boundary p p p

        lattice diamond 5.43
        region box block 0 4 0 4 0 4
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

        input_file = joinpath(lammps_test_dir, "test_mpi_domain.lmp")
        write(input_file, test_input)

        # Run with 8 MPI ranks (2x2x2 decomposition)
        output = try
            read(setenv(`mpirun -np 8 --oversubscribe $(lmp_exe) -in $(input_file)`, env), String)
        catch
            try
                read(setenv(`mpirun -np 8 $(lmp_exe) -in $(input_file)`, env), String)
            catch
                # Fall back to 4 ranks
                read(setenv(`mpirun -np 4 $(lmp_exe) -in $(input_file)`, env), String)
            end
        end

        @test !occursin("ERROR", output)
        @test occursin("Loop time", output) || occursin("Total wall time", output)

        # Extract energies and check conservation
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
            @test drift < 1e-2  # Energy conserved with MPI
        end
    end

    @testset "Ghost Atom Handling" begin
        # Test system where atoms cross domain boundaries
        test_input = """
        units metal
        atom_style atomic
        boundary p p p

        lattice diamond 5.43
        region box block 0 3 0 3 0 3
        create_box 1 box
        create_atoms 1 box
        mass 1 28.0855

        plugin load $(plugin_path)
        pair_style ace
        pair_coeff * * $(lib_path) Si

        # Higher temperature to ensure atoms move across boundaries
        velocity all create 300.0 42
        fix nve all nve

        thermo_style custom step pe atoms
        thermo 20

        run 100
        """

        input_file = joinpath(lammps_test_dir, "test_mpi_ghost.lmp")
        write(input_file, test_input)

        output = try
            read(setenv(`mpirun -np 4 --oversubscribe $(lmp_exe) -in $(input_file)`, env), String)
        catch
            read(setenv(`mpirun -np 4 $(lmp_exe) -in $(input_file)`, env), String)
        end

        @test !occursin("ERROR", output)
        @test !occursin("Lost atoms", output)  # No lost atoms
        @test occursin("Loop time", output)
    end
end
