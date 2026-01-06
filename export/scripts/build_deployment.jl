#=
build_deployment.jl - Create self-contained ACE potential deployment packages

This script takes a fitted ACE model and produces a deployment package containing:
- Compiled shared library (libace_<name>.so)
- Required Julia runtime libraries
- LAMMPS plugin (optional)
- Python/ASE wrapper (optional)
- Usage documentation

Usage:
    include("export/scripts/build_deployment.jl")
    build_deployment(model, "my_potential"; output_dir="deployments/")

Requirements:
    - Julia 1.12+ with juliac support
    - Fitted ACEpotentials model
=#

using JSON

"""
    build_deployment(model, name; kwargs...)

Create a self-contained deployment package from a fitted ACE model.

# Arguments
- `model`: Fitted ACE model from ACEpotentials
- `name::String`: Name for the deployment (used in filenames)

# Keyword Arguments
- `output_dir::String = "deployments/"`: Directory for output package
- `include_lammps::Bool = true`: Include LAMMPS plugin and examples
- `include_python::Bool = true`: Include Python/ASE wrapper
- `lammps_header_dir::Union{String,Nothing} = nothing`: Path to LAMMPS src/ for building plugin
- `julia_path::String = joinpath(Sys.BINDIR, "julia")`: Path to Julia executable
- `verbose::Bool = true`: Print progress messages

# Returns
- `String`: Path to the created deployment directory
"""
function build_deployment(
    model,
    name::String;
    output_dir::String = "deployments",
    include_lammps::Bool = true,
    include_python::Bool = true,
    lammps_header_dir::Union{String,Nothing} = nothing,
    julia_path::String = joinpath(Sys.BINDIR, "julia"),
    verbose::Bool = true
)
    # Resolve paths relative to this script's directory
    script_dir = @__DIR__
    export_dir = dirname(script_dir)

    # Create output directory
    deploy_dir = joinpath(output_dir, name)
    mkpath(joinpath(deploy_dir, "lib"))
    verbose && println("Creating deployment package: $deploy_dir")

    # Step 1: Export model to trim-compatible Julia code
    verbose && println("\n[1/6] Exporting model to trim-compatible code...")
    model_jl = joinpath(deploy_dir, "$(name)_model.jl")
    export_script = joinpath(export_dir, "src", "export_ace_model.jl")
    include(export_script)
    # Use invokelatest to handle world-age issues when include() defines new methods
    Base.invokelatest(export_ace_model, model, model_jl; splinify_first=true, for_library=true)
    verbose && println("  → Exported to: $model_jl")

    # Step 2: Compile with juliac --trim
    verbose && println("\n[2/6] Compiling shared library with juliac --trim...")
    lib_path = joinpath(deploy_dir, "lib", "libace_$(name).so")
    juliac_path = joinpath(dirname(Sys.BINDIR), "share", "julia", "juliac", "juliac.jl")

    compile_cmd = ```
        $julia_path --project=$(export_dir) $juliac_path
        --output-lib $lib_path
        --experimental --trim=safe --compile-ccallable
        $model_jl
    ```

    verbose && println("  Running: julia juliac.jl --output-lib ... $model_jl")
    run(compile_cmd)
    verbose && println("  → Compiled: $lib_path ($(round(filesize(lib_path)/1024/1024, digits=1)) MB)")

    # Step 3: Identify and copy required Julia runtime libraries
    verbose && println("\n[3/6] Bundling Julia runtime libraries...")
    bundle_julia_libs!(deploy_dir, lib_path, verbose)

    # Step 4: Copy LAMMPS plugin (if requested)
    if include_lammps
        verbose && println("\n[4/6] Including LAMMPS plugin...")
        lammps_dir = joinpath(deploy_dir, "lammps")
        mkpath(lammps_dir)

        # Copy plugin source
        plugin_src = joinpath(export_dir, "lammps", "plugin")
        if isdir(plugin_src)
            cp(plugin_src, joinpath(lammps_dir, "plugin"); force=true)
            verbose && println("  → Copied plugin source to: $(joinpath(lammps_dir, "plugin"))")
        end

        # Copy example
        example_src = joinpath(export_dir, "lammps", "examples", "in.ace_silicon")
        if isfile(example_src)
            example_dst = joinpath(lammps_dir, "example.lmp")
            cp(example_src, example_dst; force=true)
            # Update paths in example
            example_content = read(example_dst, String)
            # Update plugin path to point to bundled plugin
            example_content = replace(example_content,
                r"plugin\s+load\s+\S+" => "plugin load plugin/build/aceplugin.so")
            # Update pair_coeff path to point to bundled library
            example_content = replace(example_content,
                r"pair_coeff\s+\*\s+\*\s+\S+" => "pair_coeff * * ../lib/libace_$(name).so")
            write(example_dst, example_content)
            verbose && println("  → Created example: $example_dst")
        end

        # Build plugin if LAMMPS headers available
        if lammps_header_dir !== nothing && isdir(lammps_header_dir)
            verbose && println("  Building LAMMPS plugin...")
            build_lammps_plugin!(lammps_dir, lammps_header_dir, verbose)
        else
            verbose && println("  Note: Set lammps_header_dir to build aceplugin.so")
        end
    else
        verbose && println("\n[4/6] Skipping LAMMPS plugin (include_lammps=false)")
    end

    # Step 5: Create Python example (if requested)
    if include_python
        verbose && println("\n[5/6] Including Python/ASE example...")
        python_dir = joinpath(deploy_dir, "python")
        mkpath(python_dir)

        # Create example script
        create_python_example!(python_dir, name, verbose)

        # Create requirements.txt pointing to ase-ace package
        write(joinpath(python_dir, "requirements.txt"), """
            ase-ace[library]
            """)
        verbose && println("  → Created: requirements.txt")

        # Create installation instructions
        write(joinpath(python_dir, "README.md"), """
            # Python/ASE Usage

            ## Installation

            Install the ase-ace package with library support:

            ```bash
            pip install ase-ace[library]
            ```

            ## Usage

            ```python
            from ase.build import bulk
            from ase_ace import ACELibraryCalculator

            # Point to the compiled library
            calc = ACELibraryCalculator("../lib/libace_$(name).so")

            atoms = bulk('Si', 'diamond', a=5.43)
            atoms.calc = calc

            energy = atoms.get_potential_energy()
            print(f"Energy: {energy:.4f} eV")
            ```

            See `example.py` for more examples.
            """)
        verbose && println("  → Created: README.md")
    else
        verbose && println("\n[5/6] Skipping Python wrapper (include_python=false)")
    end

    # Step 6: Create metadata and README
    verbose && println("\n[6/6] Creating metadata and documentation...")
    create_metadata!(deploy_dir, model, name, verbose)
    create_readme!(deploy_dir, name, include_lammps, include_python, verbose)

    # Clean up intermediate files
    rm(model_jl; force=true)

    verbose && println("\n" * "="^60)
    verbose && println("Deployment package created: $deploy_dir")
    verbose && println("="^60)

    return deploy_dir
end

"""
Bundle required Julia runtime libraries into the deployment.
"""
function bundle_julia_libs!(deploy_dir::String, lib_path::String, verbose::Bool)
    lib_dir = joinpath(deploy_dir, "lib")
    julia_lib_dir = joinpath(dirname(Sys.BINDIR), "lib")
    julia_lib_julia_dir = joinpath(julia_lib_dir, "julia")

    # Get list of required libraries from ldd
    ldd_output = read(`ldd $lib_path`, String)

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
            dst = joinpath(lib_dir, basename(lib))
            if !isfile(dst)
                cp(lib, dst; follow_symlinks=true)
                copied += 1
            end
        end
    end

    verbose && println("  → Copied $copied Julia runtime libraries")

    # Create wrapper script that sets LD_LIBRARY_PATH
    wrapper_script = joinpath(deploy_dir, "setup_env.sh")
    write(wrapper_script, """
        #!/bin/bash
        # Source this file to set up the environment for using the ACE potential
        # Usage: source setup_env.sh

        SCRIPT_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
        export LD_LIBRARY_PATH="\$SCRIPT_DIR/lib:\$LD_LIBRARY_PATH"
        echo "ACE potential environment configured."
        echo "Library path: \$SCRIPT_DIR/lib"
        """)
    chmod(wrapper_script, 0o755)
    verbose && println("  → Created: setup_env.sh")
end

"""
Build LAMMPS plugin if headers are available.
"""
function build_lammps_plugin!(lammps_dir::String, lammps_header_dir::String, verbose::Bool)
    plugin_dir = joinpath(lammps_dir, "plugin")
    build_dir = joinpath(plugin_dir, "build")
    mkpath(build_dir)

    cmake_dir = joinpath(plugin_dir, "cmake")

    # Run cmake
    cd(build_dir) do
        run(`cmake $cmake_dir -DLAMMPS_HEADER_DIR=$lammps_header_dir`)
        run(`make -j4`)
    end

    # Copy built plugin to lammps_dir
    plugin_so = joinpath(build_dir, "aceplugin.so")
    if isfile(plugin_so)
        cp(plugin_so, joinpath(lammps_dir, "aceplugin.so"); force=true)
        verbose && println("  → Built: aceplugin.so")
    end
end

"""
Create Python example script.
"""
function create_python_example!(python_dir::String, name::String, verbose::Bool)
    example_path = joinpath(python_dir, "example.py")
    write(example_path, """
        #!/usr/bin/env python3
        \"\"\"Example usage of ACE potential with ASE.

        Prerequisites:
            pip install ase-ace[library]

        Usage:
            source ../setup_env.sh  # Set LD_LIBRARY_PATH
            python example.py
        \"\"\"

        from ase.build import bulk
        from ase_ace import ACELibraryCalculator

        # Load the ACE potential
        # Note: source setup_env.sh first, or set LD_LIBRARY_PATH
        calc = ACELibraryCalculator("../lib/libace_$(name).so")

        # Create a simple structure
        atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
        atoms.calc = calc

        # Compute energy and forces
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()

        print(f"Number of atoms: {len(atoms)}")
        print(f"Energy: {energy:.6f} eV")
        print(f"Energy per atom: {energy/len(atoms):.6f} eV/atom")
        print(f"Max force component: {abs(forces).max():.6f} eV/Å")
        print(f"Pressure: {-stress[:3].mean() * 160.21766208:.2f} GPa")

        # Optional: Run geometry optimization
        # from ase.optimize import BFGS
        # opt = BFGS(atoms)
        # opt.run(fmax=0.01)

        # Optional: Run MD
        # from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        # from ase.md.verlet import VelocityVerlet
        # from ase import units
        # MaxwellBoltzmannDistribution(atoms, temperature_K=300)
        # dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
        # dyn.run(100)
        """)
    verbose && println("  → Created: example.py")
end

"""
Create metadata.json with model properties.
"""
function create_metadata!(deploy_dir::String, model, name::String, verbose::Bool)
    # Extract model properties
    metadata = Dict{String,Any}(
        "name" => name,
        "created" => string(Dates.now()),
        "julia_version" => string(VERSION),
    )

    # Try to extract model-specific info
    try
        if hasproperty(model, :rbasis) && hasproperty(model.rbasis, :rcut)
            metadata["cutoff"] = model.rbasis.rcut
        end
        if hasproperty(model, :elements)
            metadata["elements"] = string.(model.elements)
        end
    catch
        # Model introspection failed, continue without these fields
    end

    metadata_path = joinpath(deploy_dir, "metadata.json")
    open(metadata_path, "w") do f
        JSON.print(f, metadata, 2)
    end
    verbose && println("  → Created: metadata.json")
end

"""
Create README.md with usage instructions.
"""
function create_readme!(deploy_dir::String, name::String, include_lammps::Bool, include_python::Bool, verbose::Bool)
    readme_path = joinpath(deploy_dir, "README.md")

    content = """
    # ACE Potential: $name

    This is a self-contained deployment of an ACE (Atomic Cluster Expansion) potential
    compiled from ACEpotentials.jl.

    ## Contents

    ```
    $name/
    ├── lib/
    │   ├── libace_$name.so      # Compiled ACE potential
    │   └── *.so                  # Required runtime libraries
    ├── setup_env.sh              # Environment setup script
    """

    if include_lammps
        content *= """
        ├── lammps/
        │   ├── plugin/               # LAMMPS plugin source
        │   ├── aceplugin.so          # Pre-built plugin (if available)
        │   └── example.lmp           # Example LAMMPS input
        """
    end

    if include_python
        content *= """
        ├── python/
        │   ├── requirements.txt      # Python dependencies (ase-ace[library])
        │   ├── example.py            # Example Python script
        │   └── README.md             # Installation instructions
        """
    end

    content *= """
    └── metadata.json             # Model metadata
    ```

    ## Setup

    Before using the potential, set up the library path:

    ```bash
    source setup_env.sh
    ```

    Or manually:
    ```bash
    export LD_LIBRARY_PATH=/path/to/$name/lib:\$LD_LIBRARY_PATH
    ```

    """

    if include_lammps
        content *= """
        ## LAMMPS Usage

        1. Build or use the pre-built LAMMPS plugin:
           ```bash
           cd lammps/plugin/build
           cmake ../cmake -DLAMMPS_HEADER_DIR=/path/to/lammps/src
           make
           ```

        2. In your LAMMPS input script:
           ```lammps
           plugin load /path/to/$name/lammps/aceplugin.so
           pair_style ace
           pair_coeff * * /path/to/$name/lib/libace_$name.so Si O ...
           ```

        See `lammps/example.lmp` for a complete example.

        """
    end

    if include_python
        content *= """
        ## Python/ASE Usage

        1. Install the ase-ace package:
           ```bash
           pip install ase-ace[library]
           ```

        2. Use in Python:
           ```python
           from ase.build import bulk
           from ase_ace import ACELibraryCalculator

           calc = ACELibraryCalculator("lib/libace_$name.so")
           atoms = bulk('Si', 'diamond', a=5.43)
           atoms.calc = calc

           energy = atoms.get_potential_energy()
           forces = atoms.get_forces()
           ```

        See `python/example.py` for more examples.

        """
    end

    content *= """
    ## Requirements

    - **No Julia installation required** - all runtime libraries are bundled
    - LAMMPS users: Any recent LAMMPS version with plugin support
    - Python users: Python 3.8+, numpy, ASE

    ## Generated by

    ACEpotentials.jl export feature
    https://github.com/ACEsuit/ACEpotentials.jl
    """

    write(readme_path, content)
    verbose && println("  → Created: README.md")
end

# Import Dates for metadata
using Dates
