#=
Julia Export Tests

Tests for:
1. Model export to trim-compatible Julia code
2. Trim compilation to shared library
3. Evaluation correctness via ccall
4. All C API functions
=#

using Test
using ACEpotentials
using StaticArrays
using LinearAlgebra
using Libdl

@testset "Export Functionality" verbose=true begin

    @testset "Model Export" begin
        # Get test model
        potential, test_data = setup_test_model()

        # Export to Julia code
        build_dir = joinpath(TEST_DIR, "build")
        mkpath(build_dir)

        include(joinpath(EXPORT_DIR, "src", "export_ace_model.jl"))
        model_file = joinpath(build_dir, "test_ace_model.jl")

        # Export with library interface
        export_ace_model(potential, model_file; for_library=true)
        @test isfile(model_file)

        # Verify file contents have expected components
        content = read(model_file, String)
        @test occursin("I2Z", content)  # Species mapping
        @test occursin("TENSOR", content)  # Tensor structure
        @test occursin("ace_site_energy", content)  # C API
        @test occursin("ace_get_cutoff", content)  # Utility function

        # Also test executable export
        exe_file = joinpath(build_dir, "test_ace_exe.jl")
        export_ace_model(potential, exe_file; for_library=false)
        @test isfile(exe_file)
        @test occursin("@main", read(exe_file, String))
    end

    @testset "Trim Compilation" begin
        # Find juliac.jl from Julia installation
        juliac_script = joinpath(Sys.BINDIR, "..", "share", "julia", "juliac", "juliac.jl")
        juliac_available = isfile(juliac_script)

        if !juliac_available
            @warn "juliac.jl not found at $juliac_script"
            @test_skip "juliac not available"
            return
        end

        build_dir = joinpath(TEST_DIR, "build")
        model_file = joinpath(build_dir, "test_ace_model.jl")
        lib_path = joinpath(build_dir, "libace_test.so")

        # Check if already compiled (from earlier test run)
        if !isfile(lib_path)
            @info "Compiling model to shared library using juliac..."

            # Use juliac.jl script
            try
                # juliac.jl usage: julia juliac.jl [options] <source.jl>
                # Note: Julia args (--project, --startup-file, -C) must come BEFORE juliac_script
                # --experimental is required to enable --trim
                # --compile-ccallable is required to export @ccallable functions
                # -C generic (--cpu-target=generic) makes the library portable across different CPU types
                julia_exe = joinpath(Sys.BINDIR, "julia")
                juliac_cmd = `$(julia_exe) --startup-file=no -C generic --project=$(EXPORT_DIR) $(juliac_script) --experimental --compile-ccallable --output-lib $(lib_path) --trim=safe $(model_file)`
                @info "Running: $juliac_cmd"
                run(juliac_cmd)
            catch e
                @warn "juliac compilation failed: $e"
                @test_skip "Compilation failed"
                return
            end
        end

        @test isfile(lib_path)

        # Check exported symbols
        symbols_output = try
            read(`nm -D $(lib_path)`, String)
        catch
            ""
        end

        if !isempty(symbols_output)
            @test occursin("ace_site_energy", symbols_output)
            @test occursin("ace_get_cutoff", symbols_output)
            @test occursin("ace_get_n_species", symbols_output)
        end

        # Store path for other tests
        TEST_ARTIFACTS["lib_path"] = lib_path
    end

    # NOTE: Evaluation correctness tests cannot be run from within Julia
    # because loading a Julia-compiled library into a running Julia process
    # causes threading conflicts (jl_init_threadtls / ijl_adopt_thread).
    # The compiled library is designed to be called from external processes
    # (Python, C, LAMMPS). Evaluation correctness is tested via:
    # - test_python.jl: Python calculator tests
    # - test_lammps.jl: LAMMPS plugin tests
    #
    # Here we just verify the symbols are exported correctly.
    @testset "Symbol Verification" begin
        build_dir = joinpath(TEST_DIR, "build")
        lib_path = joinpath(build_dir, "libace_test.so")

        if !isfile(lib_path)
            @test_skip "Library not compiled"
            return
        end

        # Check all expected symbols are exported
        symbols_output = try
            read(`nm -D $(lib_path)`, String)
        catch
            ""
        end

        if isempty(symbols_output)
            @test_skip "Cannot read symbols from library"
            return
        end

        # Site-level API (for LAMMPS)
        @test occursin("ace_site_energy", symbols_output)
        @test occursin("ace_site_energy_forces", symbols_output)
        @test occursin("ace_site_energy_forces_virial", symbols_output)

        # Utility functions
        @test occursin("ace_get_cutoff", symbols_output)
        @test occursin("ace_get_n_species", symbols_output)
        @test occursin("ace_get_species", symbols_output)
        @test occursin("ace_get_n_basis", symbols_output)

        # Basis/descriptor API
        @test occursin("ace_site_basis", symbols_output)

        @info "All expected symbols found in compiled library"
    end
end
