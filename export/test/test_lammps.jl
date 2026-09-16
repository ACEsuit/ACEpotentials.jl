#=
LAMMPS Plugin Tests (serial, plus a two-rank parity run)

The gate this file exists for, and WHICH REFERENCE EACH CHECK USES:

  1. `pair_style ace` in LAMMPS  vs  the exported model evaluated in Julia   **1e-10**
     (`exported_efv` from check_export.jl, on the geometry LAMMPS itself wrote out with
     `write_data`).  This is the plan's LAMMPS gate.  It replaced a comparison to Python at
     1e-6 that, on top of being 4 orders of magnitude loose, was DEAD: it ran only
     `if haskey(TEST_ARTIFACTS, "python_energy_8atom")` and nothing ever set that key.

  2. the compiled library through the Python C API  vs  the same Julia numbers   **1e-12**
     Kept as a second, independent check so that a failure of (1) is attributable: if (2)
     also fails, the exported model and the library disagree; if only (1) fails, the fault
     is on the LAMMPS side (neighbour list, pair style, MPI).

  3. two MPI ranks  vs  one rank, same geometry, same library   **1e-12**
     Delegated to `export/lammps/test/run_two_ranks.sh` so CI can run it standalone.

Both sides of (1) and (2) evaluate the *same* coordinates: LAMMPS builds the cell, perturbs
it and writes `geom.data`; Julia and Python read that file back.  The test additionally
asserts that the coordinates in `geom.data` are bit-identical to the ones in the 17-digit
dump, so "identical coordinates" is checked rather than assumed.

WHAT MODEL THESE GATES RUN ON, AND WHY IT MATTERS.  `build/test_etace_model.jl` (and the
`libace_test.so` compiled from it) is exported from a FULL `StackedCalculator` --
`ETOneBody + ETPairModel + ETACE` -- built by `setup_stacked_model` in
`test_etace_export.jl`.  It used to be exported from a bare `ETACEPotential`, whose
generated file said `# PAIR POTENTIAL: none in this model` and whose `pair_energy_d`
returned a hard `(0.0, 0.0)`.  All three gates above therefore ran on a many-body-only
model: a sign error or a wrong per-pair cutoff in the generated pair code would have left
every LAMMPS-side check green, and the only thing that would have caught it
(`test_pair_export.jl`) is Julia-only and guarded on host-local fixture data, so it never
runs on a hosted CI runner.  The gate below asserts `PAIR_C`, `E0_1` and a non-zero
`pair_energy_d` so that this cannot silently regress.

The pair coefficients here are random, not fitted -- that is fine, and is the point: the
pair *code path* is what these gates are covering.  Quantitative pair parity against a
fitted model remains `test_pair_export.jl`'s job.

Everything else here (plugin loading, stress symmetry, NVE) is a smoke test of the plugin
and is labelled as such -- the CI model has random parameters, so its energy conservation
carries no physics.
=#

using Test
using DelimitedFiles
using Statistics: std, mean
using LinearAlgebra: norm
using Printf: @sprintf

include(joinpath(@__DIR__, "check_export.jl"))

@testset "LAMMPS Plugin" verbose=true begin
    setup = lammps_setup()
    lib_path = setup.lib_path
    plugin_path = setup.plugin_path
    lmp_exe = setup.exe
    env = setup.env
    lammps_test_dir = joinpath(TEST_DIR, "lammps")
    model_file = joinpath(TEST_DIR, "build", "test_etace_model.jl")
    mkpath(lammps_test_dir)

    if !isfile(lib_path)
        @test_skip "ACE library not compiled - skipping LAMMPS tests"
        return
    end
    if isempty(lmp_exe)
        @test_skip "LAMMPS not found - skipping tests"
        return
    end
    @info "Using LAMMPS: $lmp_exe"

    # Build the plugin if it is not there yet.
    if !isfile(plugin_path)
        @info "Building LAMMPS ACE plugin..."
        cmake_dir = joinpath(EXPORT_DIR, "lammps", "plugin", "cmake")
        lammps_src = get(ENV, "LAMMPS_SRC", "")
        if isempty(lammps_src) || !isdir(lammps_src)
            for src in [joinpath(dirname(dirname(lmp_exe)), "src"),
                        joinpath(dirname(lmp_exe), "..", "src"),
                        "/usr/local/include/lammps", "/usr/include/lammps"]
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
        mkpath(dirname(plugin_path))
        cd(dirname(plugin_path)) do
            run(`cmake $(cmake_dir) -DLAMMPS_HEADER_DIR=$(lammps_src)`)
            run(`make -j4`)
        end
        if !isfile(plugin_path)
            @test_skip "Plugin build failed"
            return
        end
    end

    """
    Run a LAMMPS input under the environment `find_lammps_exe` proved the executable runs in,
    and return stdout+stderr combined.  A non-zero exit appends `LAMMPS_EXIT_NONZERO` instead
    of throwing, so a failing run shows up as a test failure with LAMMPS's own message in the
    log rather than as a bare `failed process` error.
    """
    function run_lmp(input::AbstractString, name::AbstractString)
        input_file = joinpath(lammps_test_dir, name)
        write(input_file, input)
        buf = IOBuffer()
        ok = try
            success(pipeline(setenv(`$(lmp_exe) -in $(input_file)`, env);
                             stdout = buf, stderr = buf))
        catch e
            @error "LAMMPS run failed to start" input_file exception = e
            false
        end
        out = String(take!(buf))
        ok || (out *= "\nLAMMPS_EXIT_NONZERO\n")
        return out
    end

    "The `ACE_ENERGY` line every input below prints at 17 significant digits."
    function parse_ace_energy(output)
        m = match(r"ACE_ENERGY\s+(\S+)", output)
        return m === nothing ? nothing : parse(Float64, m.captures[1])
    end

    @testset "Plugin Loading" begin
        out = run_lmp("""
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
        """, "test_load.lmp")
        @test !occursin("ERROR", out)
        @test occursin("Loop time", out) || occursin("Total wall time", out)
    end

    # =====================================================================================
    # THE GATE: LAMMPS vs the exported model evaluated in Julia, 1e-10
    # =====================================================================================
    geom_file = joinpath(lammps_test_dir, "geom.data")
    dump_file = joinpath(lammps_test_dir, "forces.dump")
    E_lmp = Ref{Union{Nothing,Float64}}(nothing)
    E_jl = Ref(0.0); F_jl = Ref(SVector{3,Float64}[]); natoms = Ref(0)

    @testset "LAMMPS vs Julia (reference: exported model in Julia, tol 1e-10)" begin
        out = run_lmp("""
        units metal
        atom_style atomic
        boundary p p p
        lattice diamond 5.43
        region box block 0 1 0 1 0 1
        create_box 1 box
        create_atoms 1 box
        mass 1 28.0855

        # Perturb, then hand the exact geometry to the Julia side through a file rather than
        # rebuilding it there: two builders agreeing is an assumption, a file is not.
        displace_atoms all random 0.01 0.01 0.01 42
        write_data $(geom_file)

        plugin load $(plugin_path)
        pair_style ace
        pair_coeff * * $(lib_path) Si

        variable e equal pe
        dump d all custom 1 $(dump_file) id type x y z fx fy fz
        dump_modify d sort id format float %.17g
        run 0
        print "ACE_ENERGY \$(v_e:%.17g)"
        """, "test_parity.lmp")

        @test !occursin("ERROR", out)
        @test isfile(geom_file)
        @test isfile(dump_file)

        E_lmp[] = parse_ace_energy(out)
        @test E_lmp[] !== nothing
        @test isfinite(E_lmp[])

        dump = read_lammps_dump(dump_file)
        sys = read_lammps_data(geom_file, (:Si,))
        natoms[] = length(sys)
        @test natoms[] == 8

        # Both sides evaluate identical coordinates -- checked, not assumed.  `write_data`
        # and the %.17g dump must round-trip to the same doubles.
        Xdata = [SVector{3,Float64}(ustrip.(u"Å", p)) for p in position(sys, :)]
        maxdx = maximum(maximum(abs.(a .- b)) for (a, b) in zip(Xdata, dump.X))
        @info "geometry round-trip: max|x_data - x_dump| = $maxdx Å"
        @test maxdx == 0.0

        ex = load_exported(model_file)
        @test ex.I2Z == [14]        # the `(:Si,)` type map above is only valid for a Si model
        rcut = ex.RCUT_MAX

        # The library model must actually CONTAIN a pair term and an E0, or all three gates
        # in this file quietly degrade into many-body-only checks.  That is not hypothetical:
        # until Task 3's fix round the library was exported from a bare ETACEPotential, whose
        # generated `pair_energy_d` returned a hard `(0.0, 0.0)`.  Asserted here, at the gate,
        # as well as at the export site in test_etace_export.jl, because it is here that a
        # regression would go unnoticed.
        @test isdefined(ex, :PAIR_C)
        @test isdefined(ex, :E0_1)
        Vpair, dVpair = Base.invokelatest(ex.pair_energy_d, 2.35, 1, 1)
        @info @sprintf("library pair term at r = 2.35 Å: V = %.6e eV, dV/dr = %.6e eV/Å",
                       Vpair, dVpair)
        @test abs(Vpair) > 1e-8     # a live pair term, not the (0.0, 0.0) stub
        @test abs(dVpair) > 1e-8
        E, F, _ = Base.invokelatest(exported_efv, ex, sys, rcut)
        E_jl[] = E; F_jl[] = F

        dE_atom = abs(E - E_lmp[]) / natoms[]
        dF = maximum(norm.(F .- dump.F))
        @info @sprintf("LAMMPS vs Julia: |dE|/atom = %.3e eV/atom, max|dF| = %.3e eV/Å (tol 1e-10)",
                       dE_atom, dF)
        @test dE_atom <= 1e-10
        @test dF <= 1e-10

        TEST_ARTIFACTS["lammps_energy_8atom"] = E_lmp[]
        TEST_ARTIFACTS["julia_energy_8atom"] = E
    end

    # =====================================================================================
    # Attribution check: the compiled library through the Python C API vs the same Julia
    # numbers, 1e-12.  Same geometry file, so a discrepancy here is the library, not LAMMPS.
    # =====================================================================================
    @testset "Python library vs Julia (reference: exported model in Julia, tol 1e-12)" begin
        # `required_check` makes an unavailable prerequisite a FAILURE, not a skip, whenever
        # ACE_REQUIRE_GROUPS names `lammps`.  Without that, a CI job that never installed
        # `ase` would report `lammps=ran` with this attribution gate silently absent.
        if required_check(check_python_available(), "lammps",
                          "python3 with numpy/ase/ase-ace is needed for the 1e-12 " *
                          "library-vs-Julia attribution gate") &&
           required_check(natoms[] > 0, "lammps", "the parity geometry was not produced")
            penv = ace_runtime_env(dirname(lib_path))
            penv["ACE_LIB_PATH"] = lib_path
            penv["ACE_GEOM"] = geom_file
            penv["ACE_TYPE_MAP"] = "1:14"
            script = joinpath(TEST_DIR, "python", "eval_library.py")
            out = try
                read(setenv(`python3 $script`, penv), String)
            catch e
                @error "python library evaluation failed" exception = e
                ""
            end
            @test !isempty(out)
            if !isempty(out)
                rows = filter(!isempty, strip.(split(out, '\n')))
                E_py = parse(Float64, rows[1])
                F_py = [SVector{3,Float64}(parse.(Float64, split(r))...) for r in rows[2:end]]
                @test length(F_py) == natoms[]
                dE_atom = abs(E_py - E_jl[]) / natoms[]
                dF = maximum(norm.(F_py .- F_jl[]))
                @info @sprintf("library vs Julia: |dE|/atom = %.3e eV/atom, max|dF| = %.3e eV/Å (tol 1e-12)",
                               dE_atom, dF)
                @test dE_atom <= 1e-12
                @test dF <= 1e-12
            end
        end
    end

    # =====================================================================================
    # Two MPI ranks vs one rank, 1e-12.  The comparison itself lives in
    # export/lammps/test/run_two_ranks.sh (+ compare_dump.py) so that CI, Task 6 and Task 8
    # can run it without Julia.
    # =====================================================================================
    @testset "two MPI ranks vs one rank (reference: the 1-rank dump, tol 1e-12)" begin
        script = joinpath(EXPORT_DIR, "lammps", "test", "run_two_ranks.sh")
        if required_check(!isempty(setup.mpirun), "lammps",
                          "no mpirun matching this LAMMPS executable, so the 1-rank vs " *
                          "2-rank gate cannot run")
            workdir = joinpath(lammps_test_dir, "two_ranks")
            cmd = `bash $script --lmp $(lmp_exe) --mpirun $(setup.mpirun) --plugin $(plugin_path) --lib $(lib_path) --workdir $workdir --tol 1e-12`
            buf = IOBuffer()
            try
                run(pipeline(setenv(cmd, env); stdout = buf, stderr = buf))
            catch e
                @error "run_two_ranks.sh failed" exception = e
            end
            out = String(take!(buf))
            println(out)
            @test occursin("TWO_RANK_PARITY PASS", out)
        end
    end

    @testset "Stress/Virial (smoke: cubic symmetry only)" begin
        out = run_lmp("""
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
        variable e equal pe
        thermo_style custom step pe pxx pyy pzz pxy pxz pyz
        run 0
        print "ACE_ENERGY \$(v_e:%.17g)"
        """, "test_stress.lmp")

        lines = split(out, "\n")
        stress = nothing
        for (i, line) in enumerate(lines)
            if occursin("Step", line) && occursin("PotEng", line) && i < length(lines)
                parts = split(strip(lines[i+1]))
                length(parts) >= 8 && (stress = [parse(Float64, parts[j]) for j in 3:8])
                break
            end
        end
        @test stress !== nothing
        @test all(isfinite.(stress))
        rel_std = std(stress[1:3]) / abs(mean(stress[1:3]))
        @test rel_std < 0.01
    end

    @testset "NVE runs (smoke: the CI model has random parameters)" begin
        out = run_lmp("""
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
        thermo_style custom step pe ke etotal
        thermo 10
        run 100
        """, "test_nve.lmp")

        energies = Float64[]
        in_thermo = false
        for line in split(out, "\n")
            if occursin("Step", line) && occursin("TotEng", line)
                in_thermo = true
                continue
            end
            if in_thermo
                parts = split(strip(line))
                if length(parts) >= 4 && tryparse(Int, parts[1]) !== nothing
                    push!(energies, parse(Float64, parts[4]))
                elseif occursin("Loop", line) || occursin("---", line)
                    break
                end
            end
        end

        @test length(energies) >= 10
        # NOT a physics gate: the CI model's coefficients are random, so conservation is not
        # expected.  These two only assert the integrator ran without blowing up.
        @test abs(energies[end] - energies[1]) < 10.0
        @test std(energies) < 5.0
    end
end
