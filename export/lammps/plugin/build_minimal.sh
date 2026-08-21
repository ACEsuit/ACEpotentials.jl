#!/bin/bash
# Build script for ACE/Minimal LAMMPS plugin
#
# This script builds the ACE/Minimal pair style plugin without CMake.
#
# Usage:
#   export LAMMPS_SRC=/path/to/lammps/src
#   ./build_minimal.sh
#
# The script will:
#   1. Find Julia installation automatically
#   2. Compile pair_ace_minimal.cpp and aceplugin_minimal.cpp
#   3. Link with libjulia.so
#   4. Create libaceminimal.so plugin

set -e  # Exit on error

echo "=========================================="
echo "Building ACE/Minimal LAMMPS Plugin"
echo "=========================================="
echo ""

# Check for LAMMPS_SRC
if [ -z "$LAMMPS_SRC" ]; then
    echo "Error: LAMMPS_SRC environment variable not set"
    echo "Please set it to your LAMMPS source directory:"
    echo "  export LAMMPS_SRC=/path/to/lammps/src"
    exit 1
fi

if [ ! -f "$LAMMPS_SRC/lammps.h" ]; then
    echo "Error: LAMMPS source not found in $LAMMPS_SRC"
    exit 1
fi

echo "LAMMPS source: $LAMMPS_SRC"

# Find Julia
JULIA=$(which julia 2>/dev/null || true)
if [ -z "$JULIA" ]; then
    echo "Error: Julia not found in PATH"
    exit 1
fi

echo "Julia executable: $JULIA"

# Get Julia directories
JULIA_INCLUDE=$(julia -e 'print(joinpath(Sys.BINDIR, "..", "include", "julia"))')
JULIA_LIB=$(julia -e 'print(joinpath(Sys.BINDIR, "..", "lib"))')

echo "Julia include: $JULIA_INCLUDE"
echo "Julia library: $JULIA_LIB"
echo ""

# Compiler settings
CXX=${CXX:-g++}
CXXFLAGS="-std=c++11 -O3 -fPIC -Wall"

echo "Compiler: $CXX"
echo "Flags: $CXXFLAGS"
echo ""

# Create lib directory
mkdir -p lib

# Compile pair_ace_minimal.cpp
echo "[1/3] Compiling pair_ace_minimal.cpp..."
$CXX $CXXFLAGS \
    -I${LAMMPS_SRC} \
    -I${JULIA_INCLUDE} \
    -c src/pair_ace_minimal.cpp \
    -o lib/pair_ace_minimal.o

echo "[2/3] Compiling aceplugin_minimal.cpp..."
$CXX $CXXFLAGS \
    -I${LAMMPS_SRC} \
    -I${JULIA_INCLUDE} \
    -c src/aceplugin_minimal.cpp \
    -o lib/aceplugin_minimal.o

# Link into shared library
echo "[3/3] Linking libaceminimal.so..."
$CXX -shared \
    lib/pair_ace_minimal.o \
    lib/aceplugin_minimal.o \
    -L${JULIA_LIB} -ljulia \
    -Wl,-rpath,${JULIA_LIB} \
    -o lib/libaceminimal.so

echo ""
echo "=========================================="
echo "✓ Build complete!"
echo "=========================================="
echo ""
echo "Plugin created: lib/libaceminimal.so"
echo ""
echo "To use in LAMMPS:"
echo ""
echo "1. Set environment variable:"
echo "   export ACE_C_INTERFACE_PATH=/path/to/ace_c_interface_minimal.jl"
echo ""
echo "2. In your LAMMPS input script:"
echo "   plugin load $(pwd)/lib/libaceminimal.so"
echo "   pair_style ace/minimal"
echo "   pair_coeff * * /path/to/model/ Si O ..."
echo ""
echo "Or copy lib/libaceminimal.so to a directory in your LD_LIBRARY_PATH"
echo ""
