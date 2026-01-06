#!/bin/bash
set -e
export LD_LIBRARY_PATH="/software/easybuild/software/GCCcore/14.3.0/lib64:$(pwd)/deployments/oldace/lib:$(pwd)/deployments/etace_spline/lib:$(pwd)/deployments/etace_poly/lib:/software/easybuild/software/OpenMPI/5.0.8-GCC-14.3.0/lib:$LD_LIBRARY_PATH"
LAMMPS="/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp"
PLUGIN="/home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/benchmark/deployments/tial_ace/lammps/plugin/build/aceplugin.so"
OLDACE="$(pwd)/deployments/oldace/lib/libace_fair_oldace.so"

# Create input file for oldace
sed -e "s|PLUGIN_PATH|$PLUGIN|g" -e "s|OLDACE_LIB_PATH|$OLDACE|g" lammps/in.fair_oldace > /tmp/in.test_oldace

echo "Running Old ACE benchmark..."
$LAMMPS -in /tmp/in.test_oldace 2>&1 | tail -50
