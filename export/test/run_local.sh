#!/bin/bash
#
# Local Test Runner for ACE Export Tests
#
# Usage:
#   ./run_local.sh              # Run all available tests
#   ./run_local.sh export       # Run only Julia export tests
#   ./run_local.sh python       # Run only Python tests
#   ./run_local.sh lammps       # Run only LAMMPS tests
#   ./run_local.sh threading    # Run only threading tests
#   ./run_local.sh mpi          # Run only MPI tests
#
# Environment variables:
#   LAMMPS_SRC    - Path to LAMMPS source (for building plugin)
#   JULIA_THREADS - Number of Julia threads (default: 4)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EXPORT_DIR")"

# Default values
JULIA_THREADS=${JULIA_THREADS:-4}
TEST_SELECTION="${1:-all}"

echo "=============================================="
echo "ACE Export Test Runner"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Export directory: $EXPORT_DIR"
echo "Test selection: $TEST_SELECTION"
echo "Julia threads: $JULIA_THREADS"
echo ""

# Check for Julia
if ! command -v julia &> /dev/null; then
    echo "ERROR: Julia not found in PATH"
    exit 1
fi

JULIA_VERSION=$(julia --version | head -1)
echo "Julia version: $JULIA_VERSION"

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "Python version: $PYTHON_VERSION"

    # Check for required packages
    if python3 -c "import numpy; import ase" 2>/dev/null; then
        echo "Python packages: numpy, ase available"
        HAS_PYTHON=1
    else
        echo "Python packages: numpy/ase NOT available"
        HAS_PYTHON=0
    fi
else
    echo "Python: NOT available"
    HAS_PYTHON=0
fi

# Check for LAMMPS
if command -v lmp &> /dev/null; then
    LMP_PATH=$(which lmp)
    echo "LAMMPS: $LMP_PATH"
    HAS_LAMMPS=1
else
    echo "LAMMPS: NOT available"
    HAS_LAMMPS=0
fi

# Check for MPI
if command -v mpirun &> /dev/null; then
    MPI_VERSION=$(mpirun --version 2>&1 | head -1)
    echo "MPI: $MPI_VERSION"
    HAS_MPI=1
else
    echo "MPI: NOT available"
    HAS_MPI=0
fi

echo ""
echo "=============================================="
echo "Running Tests"
echo "=============================================="

# Set up environment
export JULIA_NUM_THREADS=$JULIA_THREADS

# Add Julia libraries to LD_LIBRARY_PATH
JULIA_LIB_DIR=$(julia -e 'print(joinpath(Sys.BINDIR, "..", "lib"))')
export LD_LIBRARY_PATH="$JULIA_LIB_DIR:${LD_LIBRARY_PATH:-}"

# Build LAMMPS plugin if needed and LAMMPS is available
if [[ "$HAS_LAMMPS" == "1" ]] && [[ "$TEST_SELECTION" == "all" || "$TEST_SELECTION" == "lammps" || "$TEST_SELECTION" == "mpi" ]]; then
    PLUGIN_PATH="$EXPORT_DIR/lammps/plugin/build/aceplugin.so"

    if [[ ! -f "$PLUGIN_PATH" ]]; then
        echo ""
        echo "Building LAMMPS ACE plugin..."

        # Find LAMMPS headers
        if [[ -z "${LAMMPS_SRC:-}" ]]; then
            # Try to find from lmp location
            LMP_DIR=$(dirname $(dirname $(which lmp)))
            for path in "$LMP_DIR/src" "$LMP_DIR/../src" "/usr/include/lammps" "/usr/local/include/lammps"; do
                if [[ -d "$path" ]] && [[ -f "$path/lammps.h" ]]; then
                    LAMMPS_SRC="$path"
                    break
                fi
            done
        fi

        if [[ -n "${LAMMPS_SRC:-}" ]] && [[ -d "$LAMMPS_SRC" ]]; then
            echo "LAMMPS source: $LAMMPS_SRC"
            mkdir -p "$EXPORT_DIR/lammps/plugin/build"
            cd "$EXPORT_DIR/lammps/plugin/build"
            cmake ../cmake -DLAMMPS_HEADER_DIR="$LAMMPS_SRC"
            make -j$(nproc 2>/dev/null || echo 4)
            cd "$SCRIPT_DIR"
        else
            echo "WARNING: LAMMPS source not found - cannot build plugin"
            echo "Set LAMMPS_SRC environment variable to LAMMPS source directory"
        fi
    fi
fi

# Run tests
echo ""
cd "$PROJECT_DIR"

if [[ "$TEST_SELECTION" == "all" ]]; then
    echo "Running all available tests..."
    julia --threads=$JULIA_THREADS --project=export export/test/runtests.jl
elif [[ "$TEST_SELECTION" == "export" ]]; then
    echo "Running Julia export tests..."
    julia --threads=$JULIA_THREADS --project=export export/test/runtests.jl export
elif [[ "$TEST_SELECTION" == "python" ]]; then
    if [[ "$HAS_PYTHON" == "1" ]]; then
        echo "Running Python tests..."
        julia --threads=$JULIA_THREADS --project=export export/test/runtests.jl python
    else
        echo "ERROR: Python/numpy/ase not available"
        exit 1
    fi
elif [[ "$TEST_SELECTION" == "lammps" ]]; then
    if [[ "$HAS_LAMMPS" == "1" ]]; then
        echo "Running LAMMPS tests..."
        julia --threads=$JULIA_THREADS --project=export export/test/runtests.jl lammps
    else
        echo "ERROR: LAMMPS not available"
        exit 1
    fi
elif [[ "$TEST_SELECTION" == "threading" ]]; then
    echo "Running threading tests..."
    julia --threads=$JULIA_THREADS --project=export export/test/runtests.jl threading
elif [[ "$TEST_SELECTION" == "mpi" ]]; then
    if [[ "$HAS_LAMMPS" == "1" ]] && [[ "$HAS_MPI" == "1" ]]; then
        echo "Running MPI tests..."
        julia --threads=$JULIA_THREADS --project=export export/test/runtests.jl mpi
    else
        echo "ERROR: LAMMPS or MPI not available"
        exit 1
    fi
else
    echo "ERROR: Unknown test selection: $TEST_SELECTION"
    echo "Valid options: all, export, python, lammps, threading, mpi"
    exit 1
fi

echo ""
echo "=============================================="
echo "Tests completed!"
echo "=============================================="
