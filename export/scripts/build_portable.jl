#!/usr/bin/env julia
#=
build_portable.jl - Build portable ACE deployment inside manylinux container

This script is designed to run inside an Apptainer container based on
manylinux_2_28. It produces binaries with glibc 2.28 baseline for maximum
compatibility.

Usage (inside container):
    julia --project=/repo/export /repo/export/scripts/build_portable.jl config.yaml output_dir

The script:
1. Fits/loads ACE model (same as deploy_model.jl)
2. Exports to Julia code
3. Compiles with juliac --trim
4. Bundles Julia runtime libraries (from container's glibc 2.28)
5. Builds aceplugin.so (header-only, portable)
6. Creates portable tarball

Output structure:
    model_name_portable.tar.gz
    ├── lib/                    # Model and runtime libraries
    ├── plugin/                 # LAMMPS plugin + source
    ├── lammps/                 # Example input files
    ├── setup_env.sh
    └── README.md
=#

using YAML
using ACEpotentials
using ACEfit
using ExtXYZ

# Include the existing build_deployment script for reuse
include(joinpath(@__DIR__, "build_deployment.jl"))

"""
    main(config_path::String, output_dir::String)

Main entry point for portable build script.
"""
function main(config_path::String, output_dir::String=".")
    println("=" ^ 60)
    println("ACE Portable Build Script (manylinux_2_28)")
    println("=" ^ 60)

    # Verify we're in the container environment
    check_container_environment()

    # Parse config
    println("\n[1/5] Loading configuration...")
    config = YAML.load_file(config_path)
    println("  Configuration file: $config_path")

    name = config["name"]
    mode = get(config, "mode", "fit")

    println("  Name: $name")
    println("  Mode: $mode")
    println("  Output directory: $output_dir")

    # Ensure output directory exists
    mkpath(output_dir)

    # Get or fit model
    println("\n[2/5] Preparing model...")
    if mode == "fit"
        model = fit_model_from_config(config)
    elseif mode == "export"
        model = load_model_from_path(config["model_path"])
    else
        error("Unknown mode: $mode. Use 'fit' or 'export'.")
    end

    # Create deployment using existing function
    # This handles: export, juliac --trim, library bundling
    println("\n[3/5] Building deployment package...")
    deploy_dir = build_deployment(
        model, name;
        output_dir = output_dir,
        include_lammps = get(config, "include_lammps", true),
        include_python = get(config, "include_python", false),
        verbose = true
    )

    # Verify glibc baseline of generated libraries
    println("\n[4/5] Verifying portability...")
    verify_glibc_baseline(deploy_dir, name)

    # Add plugin rebuild script to the deployment
    add_plugin_rebuild_script(deploy_dir)

    # Create tarball (more portable than zip for Linux)
    println("\n[5/5] Creating portable tarball...")
    tarball_path = create_portable_tarball(deploy_dir, name, output_dir)

    # Optionally clean up
    if get(config, "cleanup", false)
        rm(deploy_dir; recursive=true)
        println("  Cleaned up: $deploy_dir")
    end

    println("\n" * "=" ^ 60)
    println("Portable build complete!")
    println("=" ^ 60)
    println("\nTarball: $tarball_path")
    println("Size: $(round(filesize(tarball_path)/1024/1024, digits=1)) MB")
    println("glibc baseline: 2.28 (compatible with RHEL 8+, Ubuntu 20.04+)")

    println("\nTo deploy on target system:")
    println("  1. tar xzf $(basename(tarball_path))")
    println("  2. cd $(name)_portable")
    println("  3. source setup_env.sh")
    println("  4. # If LAMMPS plugin doesn't load, rebuild it:")
    println("     cd plugin && ./build_plugin.sh /path/to/lammps/src")

    return tarball_path
end

"""
Check that we're running in the expected container environment.
"""
function check_container_environment()
    # Check glibc version
    glibc_version = try
        output = read(`ldd --version`, String)
        m = match(r"(\d+\.\d+)", output)
        m !== nothing ? m.captures[1] : "unknown"
    catch
        "unknown"
    end

    println("  Container glibc version: $glibc_version")

    # We expect glibc 2.28 in manylinux_2_28
    if glibc_version != "unknown" && parse(Float64, glibc_version) > 2.29
        @warn "glibc version $glibc_version is higher than expected 2.28. " *
              "Binaries may not be portable to older systems."
    end

    # Check Julia version
    println("  Julia version: $(VERSION)")

    # Check compiler
    try
        gcc_ver = read(`gcc --version`, String)
        m = match(r"gcc.*?(\d+\.\d+)", gcc_ver)
        if m !== nothing
            println("  GCC version: $(m.captures[1])")
        end
    catch
        @warn "GCC not found in PATH"
    end
end

"""
Verify that generated libraries have the expected glibc baseline.
"""
function verify_glibc_baseline(deploy_dir::String, name::String)
    lib_dir = joinpath(deploy_dir, "lib")
    model_so = joinpath(lib_dir, "libace_$(name).so")

    if !isfile(model_so)
        @warn "Model library not found: $model_so"
        return
    end

    # Check required glibc version using objdump or readelf
    try
        # Get GLIBC version requirements
        output = read(`objdump -T $model_so`, String)

        # Find GLIBC_X.Y version markers
        glibc_versions = Set{String}()
        for m in eachmatch(r"GLIBC_(\d+\.\d+)", output)
            push!(glibc_versions, m.captures[1])
        end

        if !isempty(glibc_versions)
            max_version = maximum(parse.(Float64, collect(glibc_versions)))
            println("  Model library requires GLIBC <= $max_version")

            if max_version > 2.28
                @warn "Model library requires GLIBC $max_version, " *
                      "which may limit portability to newer systems."
            else
                println("  ✓ Compatible with glibc 2.28+ systems")
            end
        end
    catch e
        @warn "Could not verify glibc requirements: $e"
    end
end

"""
Add the plugin rebuild script to the deployment.
"""
function add_plugin_rebuild_script(deploy_dir::String)
    plugin_dir = joinpath(deploy_dir, "lammps", "plugin")

    if !isdir(plugin_dir)
        return  # LAMMPS not included
    end

    script_path = joinpath(plugin_dir, "build_plugin.sh")

    script_content = raw"""#!/bin/bash
# =============================================================================
# Rebuild aceplugin.so against your LAMMPS installation
# =============================================================================
#
# The pre-built aceplugin.so may not be compatible with your LAMMPS version
# due to ABI differences. Use this script to rebuild against your LAMMPS.
#
# Usage:
#   ./build_plugin.sh /path/to/lammps/src
#
# Requirements:
#   - CMake 3.10+
#   - C++17 compiler (GCC 8+, Clang 7+)
#   - LAMMPS source code (headers needed)
#
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/lammps/src"
    echo ""
    echo "Rebuild aceplugin.so against your LAMMPS installation."
    echo ""
    echo "LAMMPS_SRC should point to the 'src' directory of your LAMMPS source."
    echo "Example: $0 /home/user/lammps/src"
    exit 1
fi

LAMMPS_SRC="$1"

# Validate LAMMPS source directory
if [ ! -f "$LAMMPS_SRC/lammps.h" ]; then
    echo "ERROR: LAMMPS headers not found in: $LAMMPS_SRC"
    echo "Make sure this is the 'src' directory of your LAMMPS source."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building aceplugin.so..."
echo "  LAMMPS source: $LAMMPS_SRC"
echo "  Plugin dir: $SCRIPT_DIR"

# Create build directory
mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build"

# Configure with CMake
cmake ../cmake \
    -DLAMMPS_HEADER_DIR="$LAMMPS_SRC" \
    -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

echo ""
echo "Build complete!"
echo "Plugin: $SCRIPT_DIR/build/aceplugin.so"
echo ""
echo "To use in LAMMPS:"
echo "  plugin load $SCRIPT_DIR/build/aceplugin.so"
"""

    write(script_path, script_content)
    chmod(script_path, 0o755)
    println("  Added plugin rebuild script: $script_path")
end

"""
Create a portable tarball of the deployment.
"""
function create_portable_tarball(deploy_dir::String, name::String, output_dir::String)
    # Rename directory to indicate it's portable
    portable_name = "$(name)_portable"
    portable_dir = joinpath(dirname(deploy_dir), portable_name)

    # Rename if different
    if deploy_dir != portable_dir
        if isdir(portable_dir)
            rm(portable_dir; recursive=true)
        end
        mv(deploy_dir, portable_dir)
    end

    tarball_name = "$(portable_name).tar.gz"
    tarball_path = joinpath(output_dir, tarball_name)

    # Remove existing tarball if present
    isfile(tarball_path) && rm(tarball_path)

    # Create tarball
    println("  Creating: $tarball_path")
    cd(dirname(portable_dir)) do
        run(`tar czf $tarball_name $portable_name`)
        mv(tarball_name, tarball_path)
    end

    return tarball_path
end

# ============================================================================
# Model handling functions (copied from deploy_model.jl for standalone use)
# ============================================================================

"""
    fit_model_from_config(config::Dict)

Fit an ACE model based on configuration parameters.
"""
function fit_model_from_config(config::Dict)
    # Load data
    data_path = config["data_path"]
    println("  Loading training data from: $data_path")
    data = ExtXYZ.load(data_path)
    println("  Loaded $(length(data)) configurations")

    # Create model
    model_config = config["model"]
    println("  Creating model...")
    println("    Elements: $(model_config["elements"])")
    println("    Order: $(model_config["order"])")
    println("    Total degree: $(model_config["totaldegree"])")
    println("    Cutoff: $(model_config["rcut"]) Å")

    model = ACEpotentials.ace1_model(
        elements = Symbol.(model_config["elements"]),
        order = get(model_config, "order", 2),
        totaldegree = get(model_config, "totaldegree", 6),
        rcut = get(model_config, "rcut", 5.5),
    )

    println("  Model created successfully")

    # Prepare training parameters
    energy_key = get(config, "energy_key", "energy")
    force_key = get(config, "force_key", "forces")
    virial_key = get(config, "virial_key", "virial")

    # Fit
    solver_name = get(config, "solver", "BLR")
    println("  Fitting with solver: $solver_name")

    solver = if solver_name == "QR"
        ACEfit.QR()
    elseif solver_name == "LSQR"
        ACEfit.LSQR()
    else
        ACEfit.BLR()
    end

    # acefit! handles data conversion internally
    acefit!(data, model;
        solver = solver,
        energy_key = energy_key,
        force_key = force_key,
        virial_key = virial_key,
        smoothness = get(config, "smoothness_prior", true) ? 4 : 0,
    )

    println("  Fitting completed!")

    return model
end

"""
    load_model_from_path(model_path::String)

Load a previously saved ACE model from JSON file.
"""
function load_model_from_path(model_path::String)
    println("  Loading model from: $model_path")

    # Use ACEpotentials JSON interface
    model = ACEpotentials.load_model(model_path)

    println("  Model loaded successfully")
    return model
end

# Entry point
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("""
            Usage: julia --project=export build_portable.jl config.yaml [output_dir]

            Build a portable ACE deployment inside manylinux container.

            Arguments:
              config.yaml  - YAML configuration file
              output_dir   - Output directory (default: current directory)

            This script should be run inside the portable_build Apptainer container.
            Use build_portable.sh for the recommended workflow.
            """)
        exit(1)
    end

    config_path = ARGS[1]
    output_dir = length(ARGS) >= 2 ? ARGS[2] : "."

    main(config_path, output_dir)
end
