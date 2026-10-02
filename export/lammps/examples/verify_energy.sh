#!/bin/bash
export LAMMPS_DIR=/home/eng/essswb/lammps/lammps-22Jul2025/build
export GCC_LIB=/software/easybuild/software/GCCcore/14.3.0/lib64
export LD_LIBRARY_PATH=$GCC_LIB:$LAMMPS_DIR:$LD_LIBRARY_PATH
cd /home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/export/lammps/examples

echo "Verifying energy consistency across thread counts:"
for n in 1 2 4 8; do
  E=$(OMP_NUM_THREADS=$n $LAMMPS_DIR/lmp -in in.benchmark_omp 2>&1 | grep "Step.*PotEng" -A1 | tail -1 | awk '{print $4}')
  echo "  $n threads: PE = $E eV"
done
