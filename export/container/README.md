# Portable ACE Build Container

This directory contains an Apptainer/Singularity container definition for building
portable ACE model deployments using the `manylinux_2_28` environment.

## Overview

The standard ACE deployment creates binaries linked against the build system's
glibc version, which limits portability. This container uses `manylinux_2_28`
(based on AlmaLinux 8 with glibc 2.28) to produce binaries compatible with
a wide range of Linux systems.

### Compatibility Matrix

Binaries built with this container are compatible with:

| System | glibc | Compatible |
|--------|-------|------------|
| CentOS 7 / RHEL 7 | 2.17 | ❌ (use manylinux2014 instead) |
| RHEL 8 / AlmaLinux 8 / Rocky 8 | 2.28 | ✅ |
| Ubuntu 20.04 | 2.31 | ✅ |
| Ubuntu 22.04 | 2.35 | ✅ |
| Debian 11 | 2.31 | ✅ |
| RHEL 9 / Rocky 9 | 2.34 | ✅ |

## Quick Start

### Prerequisites

- [Apptainer](https://apptainer.org/) (or Singularity) installed
- Internet access for initial container build

### Building a Portable Deployment

1. **Use the wrapper script (recommended):**

   ```bash
   cd ACEpotentials.jl/export/scripts
   ./build_portable.sh /path/to/config.yaml /path/to/output
   ```

2. **Or build manually:**

   ```bash
   # Build the container (first time only)
   cd export/container
   apptainer build portable_build.sif portable_build.def

   # Run the build
   apptainer exec \
       --bind /path/to/repo:/repo:ro \
       --bind /path/to/output:/output \
       portable_build.sif \
       julia --project=/repo/export \
           /repo/export/scripts/build_portable.jl config.yaml /output
   ```

## Container Details

### Base Image

Uses `quay.io/pypa/manylinux_2_28_x86_64`, which provides:
- glibc 2.28
- GCC 12 toolchain
- Python manylinux compatibility

### Installed Software

- **Julia 1.12.0**: Official binary from julialang.org
- **Build tools**: cmake, make, zip, unzip
- **Compilers**: GCC 12+ (from manylinux toolchain)

### Environment Variables

When running inside the container:
```bash
PATH=/opt/julia-1.12.0/bin:$PATH
JULIA_DEPOT_PATH=/workspace/.julia:/opt/julia_depot
CC=gcc
CXX=g++
```

## Output Structure

The portable build produces a tarball with:

```
model_portable.tar.gz
├── lib/
│   ├── libace_model.so      # Compiled ACE model (glibc 2.28)
│   ├── libjulia.so.1.12     # Julia runtime
│   └── *.so                 # Other dependencies
├── plugin/
│   ├── aceplugin.so         # Pre-built LAMMPS plugin
│   ├── src/                 # Plugin source (for rebuilding)
│   ├── cmake/               # CMake configuration
│   └── build_plugin.sh      # Rebuild helper script
├── lammps/
│   └── example.lmp          # Example LAMMPS input
├── setup_env.sh             # Environment setup script
└── README.md                # Deployment documentation
```

## LAMMPS Plugin Compatibility

The LAMMPS plugin (`aceplugin.so`) may not work with all LAMMPS versions due to
ABI differences. If the pre-built plugin doesn't load:

```bash
cd plugin
./build_plugin.sh /path/to/lammps/src
```

This rebuilds the plugin against your specific LAMMPS installation.

### Requirements for Plugin Rebuild

- CMake 3.10+
- C++17 compiler (GCC 8+, Clang 7+)
- LAMMPS source code (headers)

## Troubleshooting

### Container Build Fails

1. **Check internet access**: Initial build downloads Julia and packages
2. **Check disk space**: Container and depot need ~5 GB
3. **Try with sudo**: Some systems require root for container builds
   ```bash
   sudo apptainer build portable_build.sif portable_build.def
   ```

### GLIBC Version Errors on Target

If you see errors like `GLIBC_2.29 not found`:
- The target system has older glibc than expected
- Consider using `manylinux2014` (glibc 2.17) for maximum compatibility

### Julia Package Errors Inside Container

```bash
# Clear cached packages and rebuild
rm -rf /workspace/.julia
apptainer exec portable_build.sif julia --project=/repo/export -e 'using Pkg; Pkg.instantiate()'
```

### Plugin Won't Load in LAMMPS

1. **Wrong LAMMPS version**: Rebuild plugin (see above)
2. **Missing libraries**: Check `LD_LIBRARY_PATH` includes the `lib/` directory
3. **Integer size mismatch**: Check LAMMPS was built with same integer sizes
   (default is `LAMMPS_SMALLBIG`)

## Building the Container Image

```bash
# Standard build
apptainer build portable_build.sif portable_build.def

# Build with fakeroot (if not root)
apptainer build --fakeroot portable_build.sif portable_build.def

# Build to specific location
apptainer build /path/to/portable_build.sif portable_build.def
```

## Advanced Usage

### Customizing the Container

Edit `portable_build.def` to:
- Change Julia version
- Add additional packages
- Modify environment variables

### Using Different glibc Baselines

For older systems (CentOS 7), modify the `From:` line:
```singularity
# For glibc 2.17 (CentOS 7+)
From: quay.io/pypa/manylinux2014_x86_64

# For glibc 2.34 (RHEL 9+, smaller images)
From: quay.io/pypa/manylinux_2_34_x86_64
```

### CI Integration

Example GitHub Actions workflow:

```yaml
jobs:
  build-portable:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Apptainer
        uses: eWaterCycle/setup-apptainer@v2

      - name: Build container
        run: |
          cd export/container
          apptainer build portable_build.sif portable_build.def

      - name: Build portable deployment
        run: |
          export/scripts/build_portable.sh config.yaml dist/

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: portable-deployment
          path: dist/*.tar.gz
```

## References

- [manylinux specification](https://github.com/pypa/manylinux)
- [Apptainer documentation](https://apptainer.org/docs/)
- [Julia downloads](https://julialang.org/downloads/)
- [ACEpotentials.jl documentation](https://acesuit.github.io/ACEpotentials.jl/)
