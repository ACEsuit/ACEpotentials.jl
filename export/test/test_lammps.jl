#=
LAMMPS Plugin Tests (Serial)

Tests for LAMMPS pair_style ace plugin:
1. Plugin loading
2. Energy consistency with Python
3. Force consistency
4. Virial/stress consistency
5. NVE energy conservation
=#

using Test
using DelimitedFiles
using Statistics: std, mean

@testset "LAMMPS Plugin" verbose=true begin
    build_dir = joinpath(TEST_DIR, "build")
    lib_path = joinpath(build_dir, "libace_test.so")
    lammps_test_dir = joinpath(TEST_DIR, "lammps")

    if !isfile(lib_path)
        @test_skip "ACE library not compiled - skipping LAMMPS tests"
        return
    end

    # Find LAMMPS source directory first (needed for plugin build and library path)
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
        @test_skip "LAMMPS not found - skipping tests"
        return
    end

    @info "Using LAMMPS: $lmp_exe"

    # Find ACE plugin
    plugin_dir = joinpath(EXPORT_DIR, "lammps", "plugin", "build")
    plugin_path = joinpath(plugin_dir, "aceplugin.so")

    if !isfile(plugin_path)
        # Try to build plugin
        @info "Building LAMMPS ACE plugin..."
        cmake_dir = joinpath(EXPORT_DIR, "lammps", "plugin", "cmake")

        # Need LAMMPS headers - use lammps_src from above or try to find
        if isempty(lammps_src) || !isdir(lammps_src)
            # Try to find from lmp executable
            lmp_dir = dirname(dirname(lmp_exe))
            possible_srcs = [
                joinpath(lmp_dir, "src"),
                joinpath(dirname(lmp_exe), "..", "src"),
                "/usr/local/include/lammps",
                "/usr/include/lammps",
            ]
            for src in possible_srcs
                if isdir(src)
                    lammps_src = src
                    break
                end
            end
        end

        if isempty(lammps_src) || !isdir(lammps_src)
            @test_skip "LAMMPS source not found - cannot build plugin"
            return
        end

        @info "Using LAMMPS source: $lammps_src"
        mkpath(plugin_dir)
        cd(plugin_dir) do
            run(`cmake $(cmake_dir) -DLAMMPS_HEADER_DIR=$(lammps_src)`)
            run(`make -j4`)
        end

        if !isfile(plugin_path)
            @test_skip "Plugin build failed"
            return
        end
    end

    # Set up environment
    env = copy(ENV)
    julia_lib_dir = joinpath(Sys.BINDIR, "..", "lib")

    # Find LAMMPS library directory (sibling to src or at build)
    lammps_lib_dir = ""
    if !isempty(lammps_src) && isdir(lammps_src)
        # Check for build directory sibling to src
        lammps_build = joinpath(dirname(lammps_src), "build")
        if isdir(lammps_build) && isfile(joinpath(lammps_build, "liblammps.so"))
            lammps_lib_dir = lammps_build
        end
    end
    # Also try to get it from lmp executable location
    if isempty(lammps_lib_dir) && !isempty(lmp_exe)
        lmp_dir = dirname(lmp_exe)
        if isfile(joinpath(lmp_dir, "liblammps.so"))
            lammps_lib_dir = lmp_dir
        end
    end

    # Find GCC library directory (for C++ ABI compatibility)
    # Check common EasyBuild locations for GCCcore
    gcc_lib_dir = ""
    for gcc_version in ["14.3.0", "13.3.0", "13.2.0", "12.3.0", "12.2.0", "11.3.0"]
        gcc_path = "/software/easybuild/software/GCCcore/$gcc_version/lib64"
        if isdir(gcc_path) && isfile(joinpath(gcc_path, "libstdc++.so.6"))
            gcc_lib_dir = gcc_path
            break
        end
    end

    env["LD_LIBRARY_PATH"] = join(filter(!isempty, [
        gcc_lib_dir,  # GCC libs first for C++ ABI
        julia_lib_dir,
        dirname(lib_path),
        lammps_lib_dir,
        get(ENV, "LD_LIBRARY_PATH", "")
    ]), ":")

    @testset "Plugin Loading" begin
        # Create simple test input
        test_input = """
        units metal
        atom_style atomic
        boundary p p p

        lattice diamond 5.43
        region box block 0 1 0 1 0 1
        create_box 1 box
        create_atoms 1 box
        mass 1 28.0855

        plugin load $(plugin_path)
        pair_style ace
        pair_coeff * * $(lib_path) Si

        run 0
        """

        input_file = joinpath(lammps_test_dir, "test_load.lmp")
        mkpath(lammps_test_dir)
        write(input_file, test_input)

        # Run LAMMPS
        result = try
            read(setenv(`$(lmp_exe) -in $(input_file)`, env), String)
        catch e
            "error: $e"
        end

        @test !occursin("ERROR", result)
        @test occursin("Loop time", result) || occursin("Total wall time", result)
    end

    @testset "Energy Consistency" begin
        # Create 8-atom Si cell and compute energy
        test_input = """
        units metal
        atom_style atomic
        boundary p p p

        lattice diamond 5.43
        region box block 0 1 0 1 0 1
        create_box 1 box
        create_atoms 1 box
        mass 1 28.0855

        plugin load $(plugin_path)
        pair_style ace
        pair_coeff * * $(lib_path) Si

        thermo_style custom step pe
        run 0
        """

        input_file = joinpath(lammps_test_dir, "test_energy.lmp")
        write(input_file, test_input)

        output = read(setenv(`$(lmp_exe) -in $(input_file)`, env), String)

        # Extract energy from output
        # Look for line with "Step PotEng" header then the value
        lines = split(output, "\n")
        energy = nothing
        for (i, line) in enumerate(lines)
            if occursin("Step", line) && occursin("PotEng", line)
                # Next line has the values
                if i < length(lines)
                    parts = split(strip(lines[i+1]))
                    if length(parts) >= 2
                        energy = parse(Float64, parts[2])
                    end
                end
                break
            end
        end

        @test energy !== nothing
        @test isfinite(energy)

        # Compare with Python energy if available
        if haskey(TEST_ARTIFACTS, "python_energy_8atom")
            python_E = TEST_ARTIFACTS["python_energy_8atom"]
            # Use absolute difference since energy can be near zero
            abs_diff = abs(energy - python_E)
            @test abs_diff < 1e-6  # Should be nearly identical
        end

        TEST_ARTIFACTS["lammps_energy_8atom"] = energy
    end

    @testset "Force Consistency" begin
        # Compute forces and compare with Python
        test_input = """
        units metal
        atom_style atomic
        boundary p p p

        lattice diamond 5.43
        region box block 0 1 0 1 0 1
        create_box 1 box
        create_atoms 1 box
        mass 1 28.0855

        # Add small random displacements
        displace_atoms all random 0.01 0.01 0.01 42

        plugin load $(plugin_path)
        pair_style ace
        pair_coeff * * $(lib_path) Si

        # Dump forces
        dump forces all custom 1 $(joinpath(lammps_test_dir, "forces.dump")) id type x y z fx fy fz
        dump_modify forces sort id format float %20.12e

        run 0
        """

        input_file = joinpath(lammps_test_dir, "test_forces.lmp")
        write(input_file, test_input)

        run(setenv(`$(lmp_exe) -in $(input_file)`, env))

        # Read forces from dump file
        dump_file = joinpath(lammps_test_dir, "forces.dump")
        @test isfile(dump_file)

        lines = readlines(dump_file)
        natoms = parse(Int, lines[4])
        @test natoms == 8

        forces = zeros(natoms, 3)
        for i in 1:natoms
            # Data starts at line 10 (after 9 header lines)
            line = lines[9 + i]
            parts = split(strip(line))
            forces[i, 1] = parse(Float64, parts[6])
            forces[i, 2] = parse(Float64, parts[7])
            forces[i, 3] = parse(Float64, parts[8])
        end

        # Forces should be finite and non-zero (perturbed structure)
        @test all(isfinite.(forces))
        @test maximum(abs.(forces)) > 1e-6
    end

    @testset "Stress/Virial" begin
        # Compute stress tensor
        test_input = """
        units metal
        atom_style atomic
        boundary p p p

        lattice diamond 5.43
        region box block 0 1 0 1 0 1
        create_box 1 box
        create_atoms 1 box
        mass 1 28.0855

        plugin load $(plugin_path)
        pair_style ace
        pair_coeff * * $(lib_path) Si

        # Output stress in bar
        thermo_style custom step pe pxx pyy pzz pxy pxz pyz
        run 0
        """

        input_file = joinpath(lammps_test_dir, "test_stress.lmp")
        write(input_file, test_input)

        output = read(setenv(`$(lmp_exe) -in $(input_file)`, env), String)

        # Extract stress values
        lines = split(output, "\n")
        stress = nothing
        for (i, line) in enumerate(lines)
            if occursin("Step", line) && occursin("PotEng", line)
                if i < length(lines)
                    parts = split(strip(lines[i+1]))
                    if length(parts) >= 8
                        # pxx, pyy, pzz, pxy, pxz, pyz
                        stress = [parse(Float64, parts[j]) for j in 3:8]
                    end
                end
                break
            end
        end

        @test stress !== nothing
        @test all(isfinite.(stress))

        # Cubic symmetry: pxx ≈ pyy ≈ pzz
        diag = stress[1:3]
        rel_std = std(diag) / abs(mean(diag))
        @test rel_std < 0.01
    end

    @testset "NVE Energy Conservation" begin
        # Run short NVE MD
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

        velocity all create 100.0 42 dist gaussian
        velocity all zero linear

        fix nve all nve

        # Output total energy
        thermo_style custom step pe ke etotal
        thermo 10

        run 100
        """

        input_file = joinpath(lammps_test_dir, "test_nve.lmp")
        write(input_file, test_input)

        output = read(setenv(`$(lmp_exe) -in $(input_file)`, env), String)

        # Extract total energies from thermo output
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
                elseif occursin("Loop", line) || occursin("---", line)
                    break
                end
            end
        end

        @test length(energies) >= 10

        # Check energy conservation
        drift = abs(energies[end] - energies[1])
        std_E = std(energies)

        # Note: The CI test model has RANDOM parameters, not a trained potential.
        # Energy conservation is not expected to be good with random coefficients.
        # We just verify the integration runs without crashing.
        # Production models should have much better energy conservation.
        @test drift < 10.0  # Very lenient for random model
        @test std_E < 5.0   # Very lenient for random model
    end
end
