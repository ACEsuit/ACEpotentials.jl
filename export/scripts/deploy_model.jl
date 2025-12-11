#!/usr/bin/env julia
#=
deploy_model.jl - Create shareable ACE model deployment packages

Usage:
    julia --project=export export/scripts/deploy_model.jl config.yaml

Creates a self-contained zip file that can be shared with collaborators
who don't have Julia installed. The zip contains:
- Compiled ACE potential (.so file)
- Required Julia runtime libraries
- LAMMPS plugin source and example input file
- Environment setup script
- Documentation

Configuration file format (YAML):

Mode 1 - Fit from data:
```yaml
name: my_potential
mode: fit
data_path: /path/to/training.xyz
energy_key: dft_energy
force_key: dft_force
virial_key: dft_virial
model:
  elements: [Si]
  order: 2
  totaldegree: 6
  rcut: 5.5
solver: BLR
output_dir: ./deployments
```

Mode 2 - Export existing model:
```yaml
name: my_potential
mode: export
model_path: /path/to/fitted_model.json
output_dir: ./deployments
```
=#

using YAML
using ACEpotentials
using ACEfit
using ExtXYZ

# Include the existing build_deployment script
include(joinpath(@__DIR__, "build_deployment.jl"))

"""
    main(config_path::String)

Main entry point for the deployment script.
"""
function main(config_path::String)
    println("=" ^ 60)
    println("ACE Model Deployment Script")
    println("=" ^ 60)

    # Parse config
    println("\n[1/4] Loading configuration...")
    config = YAML.load_file(config_path)
    println("  Configuration file: $config_path")

    name = config["name"]
    mode = get(config, "mode", "fit")
    output_dir = get(config, "output_dir", "deployments")

    println("  Name: $name")
    println("  Mode: $mode")
    println("  Output directory: $output_dir")

    # Ensure output directory exists
    mkpath(output_dir)

    # Get or fit model
    println("\n[2/4] Preparing model...")
    if mode == "fit"
        model = fit_model_from_config(config)
    elseif mode == "export"
        model = load_model_from_path(config["model_path"])
    else
        error("Unknown mode: $mode. Use 'fit' or 'export'.")
    end

    # Create deployment using existing function
    println("\n[3/4] Building deployment package...")
    deploy_dir = build_deployment(
        model, name;
        output_dir = output_dir,
        include_lammps = get(config, "include_lammps", true),
        include_python = get(config, "include_python", false),
        verbose = true
    )

    # Create zip archive
    println("\n[4/4] Creating zip archive...")
    zip_path = create_zip_archive(deploy_dir, name, output_dir)

    # Optionally clean up unzipped directory
    if get(config, "cleanup", false)
        rm(deploy_dir; recursive=true)
        println("  Cleaned up: $deploy_dir")
    end

    println("\n" * "=" ^ 60)
    println("Deployment complete!")
    println("=" ^ 60)
    println("\nZip archive: $zip_path")
    println("Size: $(round(filesize(zip_path)/1024/1024, digits=1)) MB")
    println("\nTo use:")
    println("  1. unzip $(basename(zip_path))")
    println("  2. cd $name")
    println("  3. source setup_env.sh")
    println("  4. Run LAMMPS with: lmp -in lammps/example.lmp")

    return zip_path
end

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

"""
    create_zip_archive(deploy_dir::String, name::String, output_dir::String)

Create a zip archive of the deployment directory.
"""
function create_zip_archive(deploy_dir::String, name::String, output_dir::String)
    zip_path = joinpath(output_dir, "$(name).zip")

    # Remove existing zip if present
    isfile(zip_path) && rm(zip_path)

    # Use zip command (available on most systems)
    println("  Creating: $zip_path")
    cd(output_dir) do
        run(`zip -rq $(name).zip $(name)`)
    end

    return zip_path
end

# Entry point
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("""
            Usage: julia --project=export deploy_model.jl config.yaml

            Create a shareable ACE model deployment package.

            See export/scripts/example_deploy_config.yaml for configuration format.
            """)
        exit(1)
    end
    main(ARGS[1])
end
