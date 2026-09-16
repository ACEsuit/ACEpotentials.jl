#=
MPI Parallelization Tests

Tests for LAMMPS MPI parallel execution:
1. Domain decomposition correctness
2. Energy/force consistency between serial and parallel
3. Ghost atom handling

TWO THINGS EVERY COMPARISON IN THIS FILE HAS TO GUARD AGAINST, both of which used to be
unguarded here (and both of which only became visible when Task 3 made this group run at
all -- `which mpirun` found nothing on the development host, so it had been skipping
silently behind a one-line `@warn`):

  * THE RANKS MUST BE REAL.  An `mpirun` from a different MPI installation than the one
    `lmp` is linked against launches N processes each with an `MPI_COMM_WORLD` of size 1.
    The "parallel" run is then N independent serial runs, and every serial-vs-parallel
    comparison passes while proving nothing.  `assert_ranks` below reads the rank count out
    of LAMMPS' own `Loop time of ... on N procs` line.

  * THE FORCES MUST BE NON-TRIVIAL.  `displace_atoms all move` is a RIGID TRANSLATION of a
    perfect diamond cell, so the forces stay zero by symmetry -- measured max|F| = 2.3e-14,
    i.e. below the 2.8e-14 serial-vs-MPI disagreement it was being compared at.  A
    ghost-atom bug produces identical zeros on both sides and passes.  The geometry is now
    perturbed randomly ONCE, serially, written with `write_data` and `read_data` back by
    both runs (the pattern `export/lammps/test/run_two_ranks.sh` already uses, and the
    reason the old comment's "because random displacements differ in atom ordering between
    serial/MPI" no longer applies), and max|F| is asserted before the difference is.
=#

using Test
using LinearAlgebra: norm
using Printf: @sprintf

@testset "MPI Parallelization" verbose=true begin
    # Discovery is shared with the serial LAMMPS group: the executable has been PROVEN to
    # run, the environment is the one it ran under, and `mpirun` belongs to the same MPI
    # installation the executable is linked against (see lammps_harness.jl, K2).
    setup = lammps_setup()
    lib_path = setup.lib_path
    plugin_path = setup.plugin_path
    lmp_exe = setup.exe
    mpirun_exe = setup.mpirun
    env = setup.env
    lammps_test_dir = joinpath(TEST_DIR, "lammps")

    if !isfile(lib_path)
        @test_skip "ACE library not compiled"
        return
    end
    if !isfile(plugin_path)
        @test_skip "LAMMPS plugin not built"
        return
    end
    if isempty(lmp_exe)
        @test_skip "LAMMPS not found"
        return
    end
    # This is unreachable today: runtests.jl only includes this file when
    # `check_mpi_available() && check_lammps_available()` already held (see the `mpi` group
    # in runtests.jl's `main()`).  It is routed through `required_check` anyway, rather than
    # left as a plain `@test_skip`+`return`, because that is exactly the latent pattern
    # `required_check` exists to remove: if this file is ever `include`d some other way (a
    # future direct-invocation entry point, a different selection path), an absent `mpirun`
    # must become a FAILURE under `ACE_REQUIRE_GROUPS=mpi`, not a silent skip that lets
    # `mpi=ran` be reported with nothing underneath it.
    if !required_check(!isempty(mpirun_exe), "mpi",
                       "no mpirun matching this LAMMPS executable was found")
        return
    end

    @info "Using LAMMPS: $lmp_exe with $mpirun_exe"
    mkpath(lammps_test_dir)

    """
    Assert that LAMMPS really ran on `n` MPI ranks, from its own
    `Loop time of <t> on <n> procs for ...` line.  See the file header for why.
    """
    function assert_ranks(output::AbstractString, n::Int)
        m = match(r"Loop time of \S+ on (\d+) procs", output)
        @test m !== nothing
        m === nothing && return false
        got = parse(Int, m.captures[1])
        if got != n
            @error "LAMMPS reports a different rank count than mpirun was asked for" requested=n reported=got
        end
        @test got == n
        return got == n
    end

    """
    Run `input` on `np` ranks and return its output.  `np == 1` runs the executable directly.
    """
    function run_np(input::AbstractString, name::AbstractString, np::Int)
        f = joinpath(lammps_test_dir, name)
        write(f, input)
        np == 1 && return read(setenv(`$(lmp_exe) -in $f`, env), String)
        return try
            read(setenv(`$(mpirun_exe) -np $np --oversubscribe $(lmp_exe) -in $f`, env), String)
        catch
            read(setenv(`$(mpirun_exe) -np $np $(lmp_exe) -in $f`, env), String)
        end
    end

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

        output_serial = run_np(test_input, "test_mpi_energy.lmp", 1)
        output_mpi = run_np(test_input, "test_mpi_energy.lmp", 4)
        # Without this, a foreign mpirun giving every rank a COMM_WORLD of size 1 would make
        # the comparison below a serial-vs-serial one that always passes.  See header.
        assert_ranks(output_serial, 1)
        assert_ranks(output_mpi, 4)

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

    @testset "MPI Force Consistency (4 ranks vs serial, non-trivial forces)" begin
        # Build and perturb the cell ONCE, serially, and hand both runs the same file.
        # `displace_atoms ... random` does not reproduce under a different domain
        # decomposition, and `displace_atoms ... move` -- what this test used to do -- is a
        # rigid translation of a perfect crystal, so it compared machine zeros.  See header.
        geom = joinpath(lammps_test_dir, "mpi_geom.data")
        build_out = run_np("""
        units metal
        atom_style atomic
        boundary p p p
        lattice diamond 5.43
        region box block 0 2 0 2 0 2
        create_box 1 box
        create_atoms 1 box
        mass 1 28.0855
        displace_atoms all random 0.05 0.05 0.05 4242
        write_data $(geom)
        """, "test_mpi_geom.lmp", 1)
        @test isfile(geom)

        force_input(dumpfile) = """
        units metal
        atom_style atomic
        boundary p p p
        read_data $(geom)

        plugin load $(plugin_path)
        pair_style ace
        pair_coeff * * $(lib_path) Si

        dump forces all custom 1 $(dumpfile) id type fx fy fz
        dump_modify forces sort id format float %.17g

        run 0
        """

        dump_serial = joinpath(lammps_test_dir, "forces_serial.dump")
        dump_mpi = joinpath(lammps_test_dir, "forces_mpi.dump")
        out_serial = run_np(force_input(dump_serial), "test_mpi_forces_serial.lmp", 1)
        out_mpi = run_np(force_input(dump_mpi), "test_mpi_forces_mpi.lmp", 4)
        assert_ranks(out_serial, 1)
        assert_ranks(out_mpi, 4)

        F_serial = read_lammps_dump(dump_serial).F
        F_mpi = read_lammps_dump(dump_mpi).F

        # The compared quantity must be non-trivial before its difference means anything.
        fmax = maximum(norm.(F_serial))
        max_diff = maximum(norm.(F_serial .- F_mpi))
        @info @sprintf("MPI force consistency: max|F| = %.3e eV/Å, max|dF| (4 ranks vs serial) = %.3e eV/Å",
                       fmax, max_diff)
        @test fmax > 1.0            # a perturbed cell under a random model: ~5e1 eV/Å here
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

        # 8 ranks (2x2x2); fall back to 4 where the host cannot oversubscribe that far, and
        # assert whichever count actually materialised rather than assuming it.
        output = ""
        nranks = 0
        for np in (8, 4)
            output = try
                run_np(test_input, "test_mpi_domain.lmp", np)
            catch
                ""
            end
            if occursin("Loop time", output)
                nranks = np
                break
            end
        end
        @test nranks > 1
        nranks > 1 && assert_ranks(output, nranks)

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
            # Note: CI test model has RANDOM parameters, not a trained potential.
            # Energy conservation is not expected. Just verify MPI works.
            @test drift < 50.0  # Very lenient for random model
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

        output = run_np(test_input, "test_mpi_ghost.lmp", 4)
        assert_ranks(output, 4)

        @test !occursin("ERROR", output)
        @test !occursin("Lost atoms", output)  # No lost atoms
        @test occursin("Loop time", output)
    end
end
