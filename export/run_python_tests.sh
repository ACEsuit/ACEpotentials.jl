#!/bin/bash
source /home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/export/.venv/bin/activate
export LD_LIBRARY_PATH=/software/easybuild/software/GCCcore/14.3.0/lib64:/home/eng/essswb/lammps/lammps-22Jul2025/build:$LD_LIBRARY_PATH
export PATH=/home/eng/essswb/lammps/lammps-22Jul2025/build:$PATH

cd /home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl/export/test/python
pytest -v test_calculator.py 2>&1
